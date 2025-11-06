#!/usr/bin/env python3
"""
EBCC Decompression: HDF5 to TIFF Sequence

This script reads a compressed HDF5 file created by compress_tiff_sequence.py
and writes the decompressed frames back to TIFF files in a streaming, parallel fashion.
It preserves the original naming and copies any .log files from the source directory.
"""

import os
import sys

# Set up HDF5 plugin path for EBCC filter - MUST be done before importing h5py
from ebcc import EBCC_FILTER_DIR
os.environ["HDF5_PLUGIN_PATH"] = EBCC_FILTER_DIR

from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import argparse
import shutil
import ctypes
from ctypes import c_size_t, c_float, c_uint8, c_void_p, POINTER
import multiprocessing as mp

from ebcc import EBCC_FILTER_PATH


# ----------------------
# Module-level helpers for parallel decoding
# ----------------------
_LIB_DECODE = None


def init_decoder_worker(lib_path):
    """Initializer for decoder worker processes: load shared lib."""
    global _LIB_DECODE
    _LIB_DECODE = ctypes.CDLL(lib_path)
    _LIB_DECODE.ebcc_decode.argtypes = (
        ctypes.POINTER(ctypes.c_uint8), 
        ctypes.c_size_t, 
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    )
    _LIB_DECODE.ebcc_decode.restype = ctypes.c_size_t
    _LIB_DECODE.free_buffer.argtypes = (ctypes.c_void_p,)
    _LIB_DECODE.free_buffer.restype = None


def decode_and_write_frame_worker(args_tuple):
    """Worker function: decode a compressed frame and write to TIFF.
    
    Args:
        args_tuple: (frame_idx, compressed_bytes, output_path, min_val, max_val, 
                     height, width, original_dtype, tiff_filename)
    
    Returns:
        frame_idx on success
    """
    global _LIB_DECODE
    
    (frame_idx, compressed_bytes, output_path, min_val, max_val, 
     height, width, original_dtype, tiff_filename) = args_tuple
    
    # Decode the compressed bytes
    comp = np.ascontiguousarray(compressed_bytes)
    comp_ptr = comp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_ptr = ctypes.POINTER(ctypes.c_float)()
    out_size = _LIB_DECODE.ebcc_decode(comp_ptr, comp.size, ctypes.byref(out_ptr))
    
    # Copy decoded data
    buf = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float * out_size)).contents
    arr = np.frombuffer(buf, dtype=np.float32).copy()
    _LIB_DECODE.free_buffer(out_ptr)
    
    # Reshape to height x width
    frame_normalized = arr.reshape((height, width))
    
    # Denormalize
    if max_val > min_val:
        frame = frame_normalized * (max_val - min_val) + min_val
    else:
        frame = frame_normalized.copy()
    
    # Convert to original dtype
    if original_dtype == 'uint16':
        frame = np.clip(frame, 0, 65535).astype(np.uint16)
    elif original_dtype == 'uint8':
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif original_dtype == 'int16':
        frame = np.clip(frame, -32768, 32767).astype(np.int16)
    else:
        frame = frame.astype(np.float32)
    
    # Write TIFF file
    output_file = output_path / tiff_filename
    img = Image.fromarray(frame)
    img.save(output_file)
    
    return frame_idx


def get_original_tiff_filenames(source_dir, start_offset=0, max_frames=None):
    """
    Get list of original TIFF filenames to preserve naming.
    
    Args:
        source_dir: Original directory containing TIFF files
        start_offset: Starting frame index used during compression
        max_frames: Maximum number of frames that were compressed
    
    Returns:
        list of filenames (not full paths)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all TIFF files
    tiff_files = sorted(list(source_path.glob('*.tif')) + list(source_path.glob('*.tiff')))
    
    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {source_dir}")
    
    # Apply same offset and limit as during compression
    if max_frames is not None:
        tiff_files = tiff_files[start_offset:start_offset+max_frames]
    else:
        tiff_files = tiff_files[start_offset:]
    
    # Return just the filenames
    return [f.name for f in tiff_files]


def copy_log_files(source_dir, output_dir):
    """
    Copy all .log files from source directory to output directory.
    
    Args:
        source_dir: Source directory containing .log files
        output_dir: Destination directory
    
    Returns:
        Number of log files copied
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    log_files = list(source_path.glob('*.log'))
    
    for log_file in log_files:
        dest_file = output_path / log_file.name
        shutil.copy2(log_file, dest_file)
        print(f"Copied log file: {log_file.name}")
    
    return len(log_files)


def main():
    parser = argparse.ArgumentParser(
        description='Decompress EBCC HDF5 file back to TIFF sequence'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input HDF5 compressed file'
    )
    parser.add_argument(
        '--source-dir',
        required=True,
        help='Original source directory with TIFF files (for naming reference)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for decompressed TIFF files (default: auto-generate from source-dir)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel decompression and writing'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help='Number of worker processes for parallel mode (default: cpu_count-1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of frames to process in each batch (default: workers * 2)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EBCC HDF5 to TIFF Decompression (Streaming Mode)")
    print("=" * 70)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Determine output directory
    source_path = Path(args.source_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate: file_3_extracted -> file_3_compressed
        source_name = source_path.name
        if source_name.endswith('_extracted'):
            compressed_name = source_name.replace('_extracted', '_compressed')
        else:
            compressed_name = source_name + '_compressed'
        output_dir = source_path.parent / compressed_name
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInput HDF5: {input_path}")
    print(f"Source directory: {source_path}")
    print(f"Output directory: {output_dir}")
    
    # Read HDF5 metadata
    print("\n1. Reading HDF5 metadata...")
    with h5py.File(input_path, 'r') as f:
        min_val = f.attrs['original_min']
        max_val = f.attrs['original_max']
        num_frames = f.attrs['num_frames']
        height = f.attrs['height']
        width = f.attrs['width']
        base_cr = f.attrs.get('base_cr', 'unknown')
        residual_mode = f.attrs.get('residual_mode', 'unknown')
        
        # Determine if this is a vlen (parallel) or filter (sequential) dataset
        is_vlen = 'compressed_vlen' in f
        dataset_name = 'compressed_vlen' if is_vlen else 'compressed'
        
        print(f"Frames: {num_frames}")
        print(f"Dimensions: {width} x {height}")
        print(f"Value range: [{min_val}, {max_val}]")
        print(f"Base CR: {base_cr}")
        print(f"Residual mode: {residual_mode}")
        print(f"Dataset type: {'parallel (vlen)' if is_vlen else 'sequential (filter)'}")
    
    # Get original TIFF filenames
    print("\n2. Getting original TIFF filenames...")
    # Note: We need to know the start_offset and max_frames used during compression
    # These should ideally be stored in HDF5 attrs, but if not, we assume defaults
    start_offset = 0  # Default - should be stored in HDF5 attrs
    max_frames = num_frames  # Use the number of frames in the HDF5 file
    
    tiff_filenames = get_original_tiff_filenames(
        args.source_dir, 
        start_offset=start_offset, 
        max_frames=max_frames
    )
    
    if len(tiff_filenames) != num_frames:
        print(f"⚠️  Warning: Found {len(tiff_filenames)} TIFF filenames but HDF5 has {num_frames} frames")
        print(f"    Will generate sequential names for missing frames")
        # Generate sequential names if mismatch
        while len(tiff_filenames) < num_frames:
            tiff_filenames.append(f"frame_{len(tiff_filenames):06d}.tif")
    
    print(f"Will write {num_frames} TIFF files")
    print(f"First file: {tiff_filenames[0]}")
    print(f"Last file: {tiff_filenames[-1]}")
    
    # Infer original dtype from value range
    if max_val <= 255 and min_val >= 0:
        original_dtype = 'uint8'
    elif max_val <= 65535 and min_val >= 0:
        original_dtype = 'uint16'
    elif max_val <= 32767 and min_val >= -32768:
        original_dtype = 'int16'
    else:
        original_dtype = 'float32'
    
    print(f"Inferred original dtype: {original_dtype}")
    
    # Decompress and write TIFF files
    print("\n3. Decompressing and writing TIFF files...")
    
    if not args.parallel or not is_vlen:
        # Sequential streaming mode
        print("Sequential mode (streaming)...")
        
        with h5py.File(input_path, 'r') as f:
            dset = f[dataset_name]
            
            if is_vlen:
                # Need to decode vlen bytes manually
                lib_main = ctypes.CDLL(EBCC_FILTER_PATH)
                lib_main.ebcc_decode.argtypes = (
                    ctypes.POINTER(ctypes.c_uint8), 
                    ctypes.c_size_t, 
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
                )
                lib_main.ebcc_decode.restype = ctypes.c_size_t
                lib_main.free_buffer.argtypes = (ctypes.c_void_p,)
                lib_main.free_buffer.restype = None
                
                for i in tqdm(range(num_frames), desc="Decompressing frames"):
                    # Decode
                    comp = np.ascontiguousarray(dset[i])
                    comp_ptr = comp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                    out_ptr = ctypes.POINTER(ctypes.c_float)()
                    out_size = lib_main.ebcc_decode(comp_ptr, comp.size, ctypes.byref(out_ptr))
                    
                    buf = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float * out_size)).contents
                    arr = np.frombuffer(buf, dtype=np.float32).copy()
                    lib_main.free_buffer(out_ptr)
                    
                    frame_normalized = arr.reshape((height, width))
                    
                    # Denormalize
                    if max_val > min_val:
                        frame = frame_normalized * (max_val - min_val) + min_val
                    else:
                        frame = frame_normalized.copy()
                    
                    # Convert to original dtype
                    if original_dtype == 'uint16':
                        frame = np.clip(frame, 0, 65535).astype(np.uint16)
                    elif original_dtype == 'uint8':
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    elif original_dtype == 'int16':
                        frame = np.clip(frame, -32768, 32767).astype(np.int16)
                    else:
                        frame = frame.astype(np.float32)
                    
                    # Write TIFF
                    output_file = output_dir / tiff_filenames[i]
                    img = Image.fromarray(frame)
                    img.save(output_file)
            else:
                # HDF5 filter automatically decompresses
                for i in tqdm(range(num_frames), desc="Writing frames"):
                    frame_normalized = dset[i]
                    
                    # Denormalize
                    if max_val > min_val:
                        frame = frame_normalized * (max_val - min_val) + min_val
                    else:
                        frame = frame_normalized.copy()
                    
                    # Convert to original dtype
                    if original_dtype == 'uint16':
                        frame = np.clip(frame, 0, 65535).astype(np.uint16)
                    elif original_dtype == 'uint8':
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    elif original_dtype == 'int16':
                        frame = np.clip(frame, -32768, 32767).astype(np.int16)
                    else:
                        frame = frame.astype(np.float32)
                    
                    # Write TIFF
                    output_file = output_dir / tiff_filenames[i]
                    img = Image.fromarray(frame)
                    img.save(output_file)
    else:
        # Parallel mode (only works with vlen dataset)
        batch_size = args.batch_size if args.batch_size else (args.workers * 2)
        print(f"Parallel mode: {args.workers} workers, batch size: {batch_size}")
        
        mp_ctx = mp.get_context('spawn')
        
        with h5py.File(input_path, 'r') as f:
            dset = f[dataset_name]
            
            with mp_ctx.Pool(
                processes=args.workers, 
                initializer=init_decoder_worker, 
                initargs=(EBCC_FILTER_PATH,)
            ) as pool:
                num_batches = (num_frames + batch_size - 1) // batch_size
                print(f"Processing {num_frames} frames in {num_batches} batches...")
                
                with tqdm(total=num_frames, desc="Decompressing & writing", unit="frame") as pbar:
                    for batch_start in range(0, num_frames, batch_size):
                        batch_end = min(batch_start + batch_size, num_frames)
                        
                        # Prepare batch tasks
                        batch_tasks = []
                        for i in range(batch_start, batch_end):
                            compressed_bytes = np.ascontiguousarray(dset[i])
                            task = (
                                i, 
                                compressed_bytes, 
                                output_dir, 
                                min_val, 
                                max_val,
                                height, 
                                width, 
                                original_dtype, 
                                tiff_filenames[i]
                            )
                            batch_tasks.append(task)
                        
                        # Process batch in parallel
                        for _ in pool.imap_unordered(decode_and_write_frame_worker, batch_tasks):
                            pbar.update(1)
    
    print(f"\n✅ Successfully wrote {num_frames} TIFF files to {output_dir}")
    
    # Copy log files
    print("\n4. Copying log files...")
    num_logs = copy_log_files(args.source_dir, output_dir)
    print(f"Copied {num_logs} log file(s)")
    
    # Summary
    print("\n" + "=" * 70)
    print("DECOMPRESSION SUMMARY")
    print("=" * 70)
    print(f"Input HDF5: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frames written: {num_frames}")
    print(f"Frame dimensions: {width} x {height}")
    print(f"Output dtype: {original_dtype}")
    print(f"Log files copied: {num_logs}")
    print("=" * 70)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
