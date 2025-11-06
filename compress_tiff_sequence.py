#!/usr/bin/env python3
"""
EBCC Compression for TIFF Image Sequences

This script loads a sequence of TIFF files, applies EBCC compression,
and evaluates compression quality using PSNR and SSIM metrics.
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

from ebcc.filter_wrapper import EBCC_Filter
from ebcc import EBCC_FILTER_PATH

# For parallel encoding via the native encoder
import ctypes
from ctypes import c_size_t, c_float, c_uint8, c_void_p, POINTER, byref
import multiprocessing as mp
import multiprocessing.pool


# ----------------------
# Module-level helpers for parallel encoding (picklable)
# ----------------------
class CodecConfig(ctypes.Structure):
    _fields_ = [
        ("dims", ctypes.c_size_t * 3),
        ("base_cr", ctypes.c_float),
        ("residual_compression_type", ctypes.c_int),
        ("residual_cr", ctypes.c_float),
        ("error", ctypes.c_float),
    ]


_LIB = None
_CFG = None


def init_worker(lib_path, dims0, dims1, dims2, base_cr, residual_type, residual_cr):
    """Initializer for worker processes: load shared lib and prepare config."""
    global _LIB, _CFG
    _LIB = ctypes.CDLL(lib_path)
    # configure function prototypes
    _LIB.ebcc_encode.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(CodecConfig), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)))
    _LIB.ebcc_encode.restype = ctypes.c_size_t
    _LIB.ebcc_decode.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
    _LIB.ebcc_decode.restype = ctypes.c_size_t
    _LIB.free_buffer.argtypes = (ctypes.c_void_p,)
    _LIB.free_buffer.restype = None

    cfg = CodecConfig()
    cfg.dims[0] = dims0
    cfg.dims[1] = dims1
    cfg.dims[2] = dims2
    cfg.base_cr = float(base_cr)
    cfg.residual_compression_type = int(residual_type)
    cfg.residual_cr = float(residual_cr)
    cfg.error = float(residual_cr)
    _CFG = cfg


def compress_frame_worker(args_tuple):
    """Worker function: compress a single frame using the loaded _LIB and _CFG.

    Returns (index, bytes)
    """
    global _LIB, _CFG
    idx, frame = args_tuple
    flat = np.ascontiguousarray(frame.flatten(), dtype=np.float32)
    data_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    out_ptr = ctypes.POINTER(ctypes.c_uint8)()
    out_size = _LIB.ebcc_encode(data_ptr, ctypes.byref(_CFG), ctypes.byref(out_ptr))
    buf = ctypes.string_at(out_ptr, out_size)
    _LIB.free_buffer(out_ptr)
    return idx, buf

# ----------------------


def get_tiff_file_list(tiff_dir, start_offset=0, max_frames=None):
    """
    Get list of TIFF files to process.
    
    Args:
        tiff_dir: Directory containing TIFF files
        start_offset: Starting frame index
        max_frames: Maximum number of frames to load (None for all)
    
    Returns:
        tuple: (tiff_files, num_frames)
    """
    tiff_path = Path(tiff_dir)
    
    if not tiff_path.exists():
        raise FileNotFoundError(f"Directory not found: {tiff_dir}")
    
    # Get all TIFF files
    tiff_files = sorted(list(tiff_path.glob('*.tif')) + list(tiff_path.glob('*.tiff')))
    
    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {tiff_dir}")
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Apply offset and limit
    if max_frames is not None:
        tiff_files = tiff_files[start_offset:start_offset+max_frames]
    else:
        tiff_files = tiff_files[start_offset:]
    
    return tiff_files, len(tiff_files)


def scan_tiff_metadata(tiff_files):
    """
    Scan TIFF files to get dimensions and value range without loading all data.
    
    Args:
        tiff_files: List of TIFF file paths
    
    Returns:
        tuple: (height, width, min_val, max_val, dtype)
    """
    # Load first image to get dimensions
    first_img = Image.open(tiff_files[0])
    first_array = np.array(first_img)
    
    if first_array.ndim == 3:
        height, width = first_array.shape[:2]
        print("Color images detected - will convert to grayscale using first channel")
    else:
        height, width = first_array.shape
    
    dtype = first_array.dtype
    
    print(f"Image dimensions: {width} x {height}")
    print(f"Original dtype: {dtype}")
    
    # Scan a subset of frames to estimate min/max
    # For large sequences, sample every N frames
    num_samples = min(len(tiff_files), 100)
    sample_step = max(1, len(tiff_files) // num_samples)
    
    global_min = float('inf')
    global_max = float('-inf')
    
    print(f"Scanning {num_samples} frames to determine value range...")
    for i in tqdm(range(0, len(tiff_files), sample_step), desc="Scanning metadata"):
        img = Image.open(tiff_files[i])
        arr = np.array(img)
        
        # Handle color images
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # Take first channel
        
        global_min = min(global_min, float(arr.min()))
        global_max = max(global_max, float(arr.max()))
    
    print(f"Value range: [{global_min}, {global_max}]")
    
    return height, width, global_min, global_max, dtype


def load_single_tiff_frame(tiff_file):
    """
    Load a single TIFF file as float32 array.
    
    Args:
        tiff_file: Path to TIFF file
    
    Returns:
        numpy array (H, W) as float32
    """
    img = Image.open(tiff_file)
    arr = np.array(img, dtype=np.float32)
    
    # Handle color images
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # Take first channel
    
    return arr


def normalize_data(data, min_val, max_val):
    """Normalize data to [0, 1] range."""
    if max_val > min_val:
        data_norm = (data - min_val) / (max_val - min_val)
    else:
        data_norm = data.copy()
    return data_norm


def denormalize_data(data_norm, min_val, max_val):
    """Denormalize data from [0, 1] back to original range."""
    if max_val > min_val:
        data = data_norm * (max_val - min_val) + min_val
    else:
        data = data_norm.copy()
    return data


def compute_psnr(original, reconstructed):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = np.max(original)
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def compute_ssim_simple(original, reconstructed):
    """
    Compute a simplified SSIM (Structural Similarity Index).
    This is a basic implementation for quick evaluation.
    """
    # Constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute means
    mu_x = np.mean(original)
    mu_y = np.mean(reconstructed)
    
    # Compute variances and covariance
    sigma_x2 = np.var(original)
    sigma_y2 = np.var(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    
    # Compute SSIM
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
    
    return ssim


def main():
    parser = argparse.ArgumentParser(
        description='Apply EBCC compression to TIFF image sequences'
    )
    parser.add_argument(
        '--tiff-dir',
        default='/Users/lfbarba/GitHub/sdate/data/ct_files/file_3_extracted',
        help='Directory containing TIFF files'
    )
    parser.add_argument(
        '--output',
        default='compressed_tiff_sequence.hdf5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--start-offset',
        type=int,
        default=150,
        help='Starting frame index'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    parser.add_argument(
        '--base-cr',
        type=int,
        default=100,
        help='Base compression ratio (higher = more compression)'
    )
    parser.add_argument(
        '--residual-mode',
        choices=['none', 'relative_error_target', 'max_error_target'],
        default='relative_error_target',
        help='Residual compression mode'
    )
    parser.add_argument(
        '--residual-target',
        type=float,
        default=0.01,
        help='Residual error target value'
    )
    parser.add_argument(
        '--compare-frames',
        type=int,
        nargs='+',
        default=[0, 75, 150, 299],
        help='Frame indices to visualize (space-separated)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel compression using the native encoder and store compressed frames as vlen bytes in HDF5'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help='Number of worker processes to use for parallel encoding (default: cpu_count-1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of frames to process in each batch for parallel mode (default: workers * 2)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EBCC TIFF Sequence Compression (Streaming Mode)")
    print("=" * 70)
    
    # Scan TIFF sequence metadata
    print("\n1. Scanning TIFF sequence...")
    tiff_files, num_frames = get_tiff_file_list(
        args.tiff_dir,
        start_offset=args.start_offset,
        max_frames=args.max_frames
    )
    print(f"Processing {num_frames} frames (offset: {args.start_offset})")
    
    height, width, min_val, max_val, original_dtype = scan_tiff_metadata(tiff_files)
    
    # Set up EBCC compression
    print("\n2. Configuring EBCC compression...")
    
    if args.residual_mode == 'none':
        residual_opt = None
        print("Residual mode: NONE (base compression only)")
    else:
        residual_opt = (args.residual_mode, args.residual_target)
        print(f"Residual mode: {args.residual_mode}")
        print(f"Residual target: {args.residual_target}")
    
    ebcc_filter = EBCC_Filter(
        base_cr=args.base_cr,
        height=height,
        width=width,
        data_dim=3,  # 3D data (frames, height, width)
        residual_opt=residual_opt,
    )
    
    print(f"Base compression ratio: {args.base_cr}")
    print(f"Chunk shape: ({num_frames}, {height}, {width})")
    print(f"Filter configuration: {dict(ebcc_filter)}")
    
    # Remove existing output file
    output_path = args.output
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")
    
    # Create HDF5 file and compress
    print("\n3. Compressing data with EBCC (streaming mode)...")

    if not args.parallel:
        # Sequential streaming path using HDF5 filter
        with h5py.File(output_path, 'w') as f:
            # Store metadata
            f.attrs['original_min'] = min_val
            f.attrs['original_max'] = max_val
            f.attrs['num_frames'] = num_frames
            f.attrs['height'] = height
            f.attrs['width'] = width
            f.attrs['base_cr'] = args.base_cr
            f.attrs['residual_mode'] = args.residual_mode
            if residual_opt:
                f.attrs['residual_target'] = args.residual_target

            # Create compressed dataset (EBCC HDF5 filter) with chunking for streaming
            dset = f.create_dataset(
                'compressed',
                shape=(num_frames, height, width),
                dtype=np.float32,
                chunks=(1, height, width),  # One frame per chunk for streaming
                **ebcc_filter
            )

            # Write data frame by frame (streaming)
            print("Writing compressed data (streaming, single-process)...")
            for i in tqdm(range(num_frames), desc="Compressing frames"):
                frame = load_single_tiff_frame(tiff_files[i])
                frame_normalized = normalize_data(frame, min_val, max_val)
                dset[i] = frame_normalized

            # Read back to verify - also streaming
            print("Reading back compressed data (streaming)...")
            data_reconstructed = np.zeros((num_frames, height, width), dtype=np.float32)
            for i in tqdm(range(num_frames), desc="Decompressing frames"):
                data_reconstructed[i] = dset[i]
    else:
        # Parallel streaming path: process frames in batches
        print(f"Parallel mode enabled: using {args.workers} workers")

        # Residual mapping
        residual_map = {
            'none': 0,
            'max_error_target': 1,
            'relative_error_target': 2
        }

        # Create output HDF5 with vlen dtype to store compressed byte blobs per frame
        with h5py.File(output_path, 'w') as f:
            f.attrs['original_min'] = min_val
            f.attrs['original_max'] = max_val
            f.attrs['num_frames'] = num_frames
            f.attrs['height'] = height
            f.attrs['width'] = width
            f.attrs['base_cr'] = args.base_cr
            f.attrs['residual_mode'] = args.residual_mode
            if residual_opt:
                f.attrs['residual_target'] = args.residual_target

            vlen_dt = h5py.vlen_dtype(np.dtype('uint8'))
            dset = f.create_dataset('compressed_vlen', shape=(num_frames,), dtype=vlen_dt)

            # Streaming batch processing with parallel compression
            batch_size = args.batch_size if args.batch_size else (args.workers * 2)
            print(f"Batch size: {batch_size} frames (keeps {args.workers} workers busy)")
            print(f"Memory footprint per batch: ~{batch_size * height * width * 4 / 1e6:.2f} MB")
            
            mp_ctx = mp.get_context('spawn')
            init_args = (EBCC_FILTER_PATH, 1, height, width, float(args.base_cr), 
                        int(residual_map.get(args.residual_mode, 2)), float(args.residual_target))
            
            with mp_ctx.Pool(processes=args.workers, initializer=init_worker, initargs=init_args) as pool:
                num_batches = (num_frames + batch_size - 1) // batch_size
                print(f"Processing {num_frames} frames in {num_batches} batches...")
                
                frames_processed = 0
                with tqdm(total=num_frames, desc="Compressing frames", unit="frame") as pbar:
                    for batch_start in range(0, num_frames, batch_size):
                        batch_end = min(batch_start + batch_size, num_frames)
                        current_batch_size = batch_end - batch_start
                        
                        # Load batch of frames (streaming - only current batch in memory)
                        batch_tasks = []
                        for i in range(batch_start, batch_end):
                            frame = load_single_tiff_frame(tiff_files[i])
                            frame_normalized = normalize_data(frame, min_val, max_val)
                            batch_tasks.append((i, frame_normalized))
                        
                        # Compress batch in parallel
                        for idx, buf in pool.imap_unordered(compress_frame_worker, batch_tasks):
                            dset[idx] = np.frombuffer(buf, dtype=np.uint8)
                            frames_processed += 1
                            pbar.update(1)

            print("Parallel compression finished; compressed frames stored in 'compressed_vlen' dataset")

        # After parallel compression, decode frames for verification (streaming)
        print("Decoding compressed frames (streaming) for verification...")
        data_reconstructed = np.zeros((num_frames, height, width), dtype=np.float32)
        # load library in main process
        lib_main = ctypes.CDLL(EBCC_FILTER_PATH)
        lib_main.ebcc_decode.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        lib_main.ebcc_decode.restype = ctypes.c_size_t
        lib_main.free_buffer.argtypes = (ctypes.c_void_p,)
        lib_main.free_buffer.restype = None
        with h5py.File(output_path, 'r') as f:
            dset_v = f['compressed_vlen']
            for i in tqdm(range(num_frames), desc="Decompressing frames"):
                comp = np.ascontiguousarray(dset_v[i])
                comp_ptr = comp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                out_ptr = ctypes.POINTER(ctypes.c_float)()
                out_size = lib_main.ebcc_decode(comp_ptr, comp.size, ctypes.byref(out_ptr))
                # out_size is number of floats
                buf = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float * out_size)).contents
                arr = np.frombuffer(buf, dtype=np.float32).copy()
                lib_main.free_buffer(out_ptr)
                # reshape to height x width
                data_reconstructed[i] = arr.reshape((height, width))
    
    print("Compression complete!")
    
    # Calculate file sizes and compression ratio
    print("\n4. Analyzing compression performance...")
    # Original size: TIFF files are uint16 (2 bytes per pixel)
    bytes_per_pixel = np.dtype(original_dtype).itemsize
    original_size_on_disk = num_frames * height * width * bytes_per_pixel
    original_size_float32 = num_frames * height * width * 4  # float32 is 4 bytes
    compressed_size = os.path.getsize(output_path)
    compression_ratio_vs_original = original_size_on_disk / compressed_size
    compression_ratio_vs_float32 = original_size_float32 / compressed_size
    space_savings_vs_original = (1 - compressed_size / original_size_on_disk) * 100
    
    print(f"Original size ({original_dtype} on disk): {original_size_on_disk / 1e6:.2f} MB")
    print(f"Original size (float32 in memory): {original_size_float32 / 1e6:.2f} MB")
    print(f"Compressed size: {compressed_size / 1e6:.2f} MB")
    print(f"Compression ratio (vs {original_dtype}): {compression_ratio_vs_original:.2f}:1")
    print(f"Compression ratio (vs float32): {compression_ratio_vs_float32:.2f}:1")
    print(f"Space savings (vs {original_dtype}): {space_savings_vs_original:.1f}%")
    
    # Compute quality metrics (streaming)
    print("\n5. Computing quality metrics (streaming)...")
    
    # We'll compute metrics frame by frame without loading all original data
    psnr_per_frame = []
    ssim_per_frame = []
    max_abs_errors = []
    
    # Denormalize reconstructed data first
    data_reconstructed_original = denormalize_data(data_reconstructed, min_val, max_val)
    
    for i in tqdm(range(num_frames), desc="Computing metrics"):
        # Load original frame
        frame_original = load_single_tiff_frame(tiff_files[i])
        frame_reconstructed = data_reconstructed_original[i]
        
        # Per-frame metrics
        frame_psnr = compute_psnr(frame_original, frame_reconstructed)
        frame_ssim = compute_ssim_simple(frame_original, frame_reconstructed)
        frame_max_error = np.max(np.abs(frame_original - frame_reconstructed))
        
        psnr_per_frame.append(frame_psnr)
        ssim_per_frame.append(frame_ssim)
        max_abs_errors.append(frame_max_error)
    
    avg_psnr = np.mean(psnr_per_frame)
    min_psnr = np.min(psnr_per_frame)
    max_psnr = np.max(psnr_per_frame)
    avg_ssim = np.mean(ssim_per_frame)
    max_abs_error = np.max(max_abs_errors)
    
    print(f"Average PSNR (per frame): {avg_psnr:.2f} dB")
    print(f"PSNR range: [{min_psnr:.2f}, {max_psnr:.2f}] dB")
    print(f"Average SSIM: {avg_ssim:.6f}")
    
    # Error analysis
    data_range = max_val - min_val
    if data_range > 0:
        rel_error = max_abs_error / data_range
        print(f"Max absolute error: {max_abs_error:.2e}")
        print(f"Max relative error: {rel_error:.6f}")
    else:
        print(f"Max absolute error: {max_abs_error:.2e}")
    
    # Bits per pixel
    original_bits_per_pixel = bytes_per_pixel * 8
    original_bits_per_pixel_float32 = 32  # In-memory representation
    compressed_bits_per_pixel = (compressed_size * 8) / (num_frames * height * width)
    
    print(f"\nBits per pixel:")
    print(f"  Original ({original_dtype} on disk): {original_bits_per_pixel:.2f}")
    print(f"  Original (float32 in memory): {original_bits_per_pixel_float32:.2f}")
    print(f"  Compressed: {compressed_bits_per_pixel:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"Source: TIFF sequence from {args.tiff_dir}")
    print(f"Frames processed: {num_frames} (offset: {args.start_offset})")
    print(f"Frame dimensions: {width} x {height}")
    print(f"Compression ratio (vs {original_dtype}): {compression_ratio_vs_original:.2f}:1")
    print(f"Compression ratio (vs float32): {compression_ratio_vs_float32:.2f}:1")
    print(f"Space savings (vs {original_dtype}): {space_savings_vs_original:.1f}%")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.6f}")
    print(f"Output file: {output_path}")
    print("=" * 70)
    
    # Optional: Visualize some frames
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print("\n6. Creating visualization...")
        
        # Select frames to visualize
        vis_frames = [i for i in args.compare_frames if i < num_frames]
        
        if len(vis_frames) > 0:
            fig, axes = plt.subplots(3, len(vis_frames), figsize=(5*len(vis_frames), 15))
            if len(vis_frames) == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle('EBCC Compression: Original vs Reconstructed vs Difference',
                        fontsize=16, fontweight='bold')
            
            for i, frame_idx in enumerate(vis_frames):
                # Load original frame for visualization
                frame_original = load_single_tiff_frame(tiff_files[frame_idx])
                frame_reconstructed = data_reconstructed_original[frame_idx]
                
                # Original
                axes[0, i].imshow(frame_original, cmap='gray')
                axes[0, i].set_title(f'Original Frame {frame_idx}')
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(frame_reconstructed, cmap='gray')
                axes[1, i].set_title(f'Reconstructed Frame {frame_idx}')
                axes[1, i].axis('off')
                
                # Difference
                diff = np.abs(frame_original - frame_reconstructed)
                im = axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title(f'Difference\nMax: {diff.max():.2e}')
                axes[2, i].axis('off')
                plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            output_fig = args.output.replace('.hdf5', '_comparison.png')
            fig.savefig(output_fig, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {output_fig}")
            plt.close()
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
