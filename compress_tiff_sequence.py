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


def load_tiff_sequence(tiff_dir, start_offset=0, max_frames=None):
    """
    Load a sequence of TIFF files into a numpy array.
    
    Args:
        tiff_dir: Directory containing TIFF files
        start_offset: Starting frame index
        max_frames: Maximum number of frames to load (None for all)
    
    Returns:
        tuple: (numpy_array, min_val, max_val, file_list)
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
    
    print(f"Loading {len(tiff_files)} frames (offset: {start_offset})")
    
    # Load first image to get dimensions
    first_img = Image.open(tiff_files[0])
    first_array = np.array(first_img)
    height, width = first_array.shape[:2]
    
    print(f"Image dimensions: {width} x {height}")
    print(f"Original dtype: {first_array.dtype}")
    
    # Initialize array
    if first_array.ndim == 2:  # Grayscale
        data = np.zeros((len(tiff_files), height, width), dtype=np.float32)
    else:  # Color - convert to grayscale
        data = np.zeros((len(tiff_files), height, width), dtype=np.float32)
        print("Color images detected - converting to grayscale")
    
    # Load all images
    global_min = float('inf')
    global_max = float('-inf')
    
    for i, tiff_file in enumerate(tqdm(tiff_files, desc="Loading TIFF files")):
        img = Image.open(tiff_file)
        arr = np.array(img)
        
        # Handle color images
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # Take first channel
        
        data[i] = arr
        global_min = min(global_min, arr.min())
        global_max = max(global_max, arr.max())
    
    print(f"Value range: [{global_min}, {global_max}]")
    print(f"Data shape: {data.shape}")
    print(f"Memory usage: {data.nbytes / 1e6:.2f} MB")
    
    return data, global_min, global_max, tiff_files


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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EBCC TIFF Sequence Compression")
    print("=" * 70)
    
    # Load TIFF sequence
    print("\n1. Loading TIFF sequence...")
    data_original, min_val, max_val, tiff_files = load_tiff_sequence(
        args.tiff_dir,
        start_offset=args.start_offset,
        max_frames=args.max_frames
    )
    
    # Normalize data
    print("\n2. Normalizing data to [0, 1] range...")
    data_normalized = normalize_data(data_original, min_val, max_val)
    print(f"Normalized range: [{data_normalized.min():.6f}, {data_normalized.max():.6f}]")
    
    # Set up EBCC compression
    print("\n3. Configuring EBCC compression...")
    
    if args.residual_mode == 'none':
        residual_opt = None
        print("Residual mode: NONE (base compression only)")
    else:
        residual_opt = (args.residual_mode, args.residual_target)
        print(f"Residual mode: {args.residual_mode}")
        print(f"Residual target: {args.residual_target}")
    
    # Data shape: (num_frames, height, width)
    num_frames, height, width = data_normalized.shape
    
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
    print("\n4. Compressing data with EBCC...")

    if not args.parallel:
        # Original (single-process) path using HDF5 filter
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

            # Create compressed dataset (EBCC HDF5 filter)
            dset = f.create_dataset('compressed', shape=data_normalized.shape, **ebcc_filter)

            # Write data
            print("Writing compressed data (single-process)...")
            dset[:] = data_normalized

            # Read back to verify
            print("Reading back compressed data...")
            data_reconstructed = dset[:]
    else:
        # Parallel path: call native ebcc_encode per-frame using ctypes in worker processes
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

            # Prepare tasks
            tasks = [(i, data_normalized[i]) for i in range(num_frames)]

            # Run pool
            mp_ctx = mp.get_context('spawn')
            init_args = (EBCC_FILTER_PATH, 1, height, width, float(args.base_cr), int(residual_map.get(args.residual_mode, 2)), float(args.residual_target))
            with mp_ctx.Pool(processes=args.workers, initializer=init_worker, initargs=init_args) as pool:
                for idx, buf in pool.imap_unordered(compress_frame_worker, tasks, chunksize=1):
                    # write compressed bytes as uint8 array into vlen dataset
                    dset[idx] = np.frombuffer(buf, dtype=np.uint8)

            # For compatibility keep a placeholder 'reconstructed' dataset by decoding one frame (optional)
            print("Parallel compression finished; compressed frames stored in 'compressed_vlen' dataset")

        # After parallel compression we can (optionally) decode all frames serially to compute metrics
        print("Decoding compressed frames (serial) for verification...")
        data_reconstructed = np.zeros_like(data_normalized, dtype=np.float32)
        # load library in main process
        lib_main = ctypes.CDLL(EBCC_FILTER_PATH)
        lib_main.ebcc_decode.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        lib_main.ebcc_decode.restype = ctypes.c_size_t
        lib_main.free_buffer.argtypes = (ctypes.c_void_p,)
        lib_main.free_buffer.restype = None
        with h5py.File(output_path, 'r') as f:
            dset_v = f['compressed_vlen']
            for i in range(num_frames):
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
    print("\n5. Analyzing compression performance...")
    # Original size: TIFF files are uint16 (2 bytes per pixel), but we load as float32 in memory
    original_size_uint16 = num_frames * height * width * 2  # 2 bytes for uint16
    original_size_float32 = data_original.nbytes  # float32 in memory
    compressed_size = os.path.getsize(output_path)
    compression_ratio_vs_uint16 = original_size_uint16 / compressed_size
    compression_ratio_vs_float32 = original_size_float32 / compressed_size
    space_savings_vs_uint16 = (1 - compressed_size / original_size_uint16) * 100
    
    print(f"Original size (uint16 on disk): {original_size_uint16 / 1e6:.2f} MB")
    print(f"Original size (float32 in memory): {original_size_float32 / 1e6:.2f} MB")
    print(f"Compressed size: {compressed_size / 1e6:.2f} MB")
    print(f"Compression ratio (vs uint16): {compression_ratio_vs_uint16:.2f}:1")
    print(f"Compression ratio (vs float32): {compression_ratio_vs_float32:.2f}:1")
    print(f"Space savings (vs uint16): {space_savings_vs_uint16:.1f}%")
    
    # Denormalize reconstructed data for quality metrics
    data_reconstructed_original = denormalize_data(data_reconstructed, min_val, max_val)
    
    # Compute quality metrics
    print("\n6. Computing quality metrics...")
    
    # Overall metrics
    psnr = compute_psnr(data_original, data_reconstructed_original)
    ssim = compute_ssim_simple(data_original, data_reconstructed_original)
    
    # Per-frame metrics
    psnr_per_frame = []
    for i in range(num_frames):
        frame_psnr = compute_psnr(data_original[i], data_reconstructed_original[i])
        psnr_per_frame.append(frame_psnr)
    
    avg_psnr = np.mean(psnr_per_frame)
    min_psnr = np.min(psnr_per_frame)
    max_psnr = np.max(psnr_per_frame)
    
    print(f"Overall PSNR: {psnr:.2f} dB")
    print(f"Average PSNR (per frame): {avg_psnr:.2f} dB")
    print(f"PSNR range: [{min_psnr:.2f}, {max_psnr:.2f}] dB")
    print(f"Overall SSIM: {ssim:.6f}")
    
    # Error analysis
    max_abs_error = np.max(np.abs(data_original - data_reconstructed_original))
    data_range = max_val - min_val
    if data_range > 0:
        rel_error = max_abs_error / data_range
        print(f"Max absolute error: {max_abs_error:.2e}")
        print(f"Max relative error: {rel_error:.6f}")
    else:
        print(f"Max absolute error: {max_abs_error:.2e}")
    
    # Bits per pixel
    original_bits_per_pixel_uint16 = 16  # Original TIFF files are uint16
    original_bits_per_pixel_float32 = 32  # In-memory representation
    compressed_bits_per_pixel = (compressed_size * 8) / (num_frames * height * width)
    
    print(f"\nBits per pixel:")
    print(f"  Original (uint16 on disk): {original_bits_per_pixel_uint16:.2f}")
    print(f"  Original (float32 in memory): {original_bits_per_pixel_float32:.2f}")
    print(f"  Compressed: {compressed_bits_per_pixel:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"Source: TIFF sequence from {args.tiff_dir}")
    print(f"Frames processed: {num_frames} (offset: {args.start_offset})")
    print(f"Frame dimensions: {width} x {height}")
    print(f"Compression ratio (vs uint16): {compression_ratio_vs_uint16:.2f}:1")
    print(f"Compression ratio (vs float32): {compression_ratio_vs_float32:.2f}:1")
    print(f"Space savings (vs uint16): {space_savings_vs_uint16:.1f}%")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.6f}")
    print(f"Output file: {output_path}")
    print("=" * 70)
    
    # Optional: Visualize some frames
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print("\n7. Creating visualization...")
        
        # Select frames to visualize
        vis_frames = [i for i in args.compare_frames if i < num_frames]
        
        if len(vis_frames) > 0:
            fig, axes = plt.subplots(3, len(vis_frames), figsize=(5*len(vis_frames), 15))
            if len(vis_frames) == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle('EBCC Compression: Original vs Reconstructed vs Difference',
                        fontsize=16, fontweight='bold')
            
            for i, frame_idx in enumerate(vis_frames):
                # Original
                axes[0, i].imshow(data_original[frame_idx], cmap='gray')
                axes[0, i].set_title(f'Original Frame {frame_idx}')
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(data_reconstructed_original[frame_idx], cmap='gray')
                axes[1, i].set_title(f'Reconstructed Frame {frame_idx}')
                axes[1, i].axis('off')
                
                # Difference
                diff = np.abs(data_original[frame_idx] - data_reconstructed_original[frame_idx])
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
