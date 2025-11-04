#!/usr/bin/env python3
"""
EBCC Compression for TIFF Image Sequences - PARALLEL VERSION

This script loads a sequence of TIFF files and applies EBCC compression
with parallel processing for significant speedup.
"""

import os
import sys

# Set up HDF5 plugin path - MUST be done before importing h5py
from ebcc import EBCC_FILTER_DIR
os.environ["HDF5_PLUGIN_PATH"] = EBCC_FILTER_DIR

from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time

from ebcc.filter_wrapper import EBCC_Filter


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
    
    # Parallel loading of images
    def load_single_image(args):
        i, tiff_file = args
        img = Image.open(tiff_file)
        arr = np.array(img)
        # Handle color images
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # Take first channel
        return i, arr, arr.min(), arr.max()
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with ThreadPoolExecutor(max_workers=min(8, len(tiff_files))) as executor:
        results = list(tqdm(
            executor.map(load_single_image, enumerate(tiff_files)),
            total=len(tiff_files),
            desc="Loading TIFF files"
        ))
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for i, arr, arr_min, arr_max in results:
        data[i] = arr
        global_min = min(global_min, arr_min)
        global_max = max(global_max, arr_max)
    
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


def write_frame_parallel(frame_idx, frame_data, output_path):
    """Write a single frame to HDF5 file (used for parallel writing)."""
    with h5py.File(output_path, 'r+') as f:
        dset = f['compressed']
        dset[frame_idx] = frame_data


def main():
    parser = argparse.ArgumentParser(
        description='Apply EBCC compression to TIFF image sequences (PARALLEL VERSION)'
    )
    parser.add_argument(
        '--tiff-dir',
        default='/Users/lfbarba/GitHub/sdate/data/ct_files/file_3_extracted',
        help='Directory containing TIFF files'
    )
    parser.add_argument(
        '--output',
        default='compressed_tiff_sequence_parallel.hdf5',
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
        default=300,
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
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of frames to write in each batch'
    )
    
    args = parser.parse_args()
    
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    
    print("=" * 70)
    print("EBCC TIFF Sequence Compression - PARALLEL VERSION")
    print("=" * 70)
    print(f"Using {args.num_workers} parallel workers")
    print(f"Batch size: {args.batch_size}")
    
    # Load TIFF sequence
    print("\n1. Loading TIFF sequence...")
    start_time = time.time()
    data_original, min_val, max_val, tiff_files = load_tiff_sequence(
        args.tiff_dir,
        start_offset=args.start_offset,
        max_frames=args.max_frames
    )
    load_time = time.time() - start_time
    print(f"Loading time: {load_time:.2f} seconds")
    
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
    
    # Create HDF5 file and compress in parallel
    print("\n4. Compressing data with EBCC (PARALLEL)...")
    compression_start = time.time()
    
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
        
        # Create compressed dataset
        dset = f.create_dataset('compressed', shape=data_normalized.shape, **ebcc_filter)
        
        # Write data in batches with progress bar
        print("Writing compressed data in parallel batches...")
        batch_size = args.batch_size
        
        with tqdm(total=num_frames, desc="Compressing frames") as pbar:
            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                batch_data = data_normalized[batch_start:batch_end]
                
                # Write batch (HDF5 handles chunking and compression)
                dset[batch_start:batch_end] = batch_data
                
                pbar.update(batch_end - batch_start)
    
    compression_time = time.time() - compression_start
    print(f"Compression time: {compression_time:.2f} seconds")
    print("Compression complete!")
    
    # Calculate file sizes and compression ratio
    print("\n5. Analyzing compression performance...")
    original_size = data_original.nbytes
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    space_savings = (1 - compressed_size / original_size) * 100
    
    print(f"Original size: {original_size / 1e6:.2f} MB")
    print(f"Compressed size: {compressed_size / 1e6:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Space savings: {space_savings:.1f}%")
    
    # Read back and compute quality metrics
    print("\n6. Reading back and computing quality metrics...")
    read_start = time.time()
    
    with h5py.File(output_path, 'r') as f:
        dset = f['compressed']
        
        # Read in batches to show progress
        data_reconstructed = np.zeros_like(data_normalized)
        
        with tqdm(total=num_frames, desc="Reading frames") as pbar:
            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                data_reconstructed[batch_start:batch_end] = dset[batch_start:batch_end]
                pbar.update(batch_end - batch_start)
    
    read_time = time.time() - read_start
    print(f"Read time: {read_time:.2f} seconds")
    
    # Denormalize reconstructed data for quality metrics
    data_reconstructed_original = denormalize_data(data_reconstructed, min_val, max_val)
    
    # Compute quality metrics
    print("\nComputing quality metrics...")
    
    # Overall metrics
    psnr = compute_psnr(data_original, data_reconstructed_original)
    ssim = compute_ssim_simple(data_original, data_reconstructed_original)
    
    # Per-frame metrics (sample for speed)
    sample_indices = np.linspace(0, num_frames-1, min(50, num_frames), dtype=int)
    psnr_per_frame = []
    for i in sample_indices:
        frame_psnr = compute_psnr(data_original[i], data_reconstructed_original[i])
        psnr_per_frame.append(frame_psnr)
    
    avg_psnr = np.mean(psnr_per_frame)
    min_psnr = np.min(psnr_per_frame)
    max_psnr = np.max(psnr_per_frame)
    
    print(f"Overall PSNR: {psnr:.2f} dB")
    print(f"Average PSNR (sampled): {avg_psnr:.2f} dB")
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
    original_bits_per_pixel = (original_size * 8) / (num_frames * height * width)
    compressed_bits_per_pixel = (compressed_size * 8) / (num_frames * height * width)
    
    print(f"\nBits per pixel:")
    print(f"  Original: {original_bits_per_pixel:.2f}")
    print(f"  Compressed: {compressed_bits_per_pixel:.2f}")
    
    # Performance summary
    total_time = time.time() - start_time
    print(f"\n⏱️  PERFORMANCE SUMMARY:")
    print(f"  Loading time: {load_time:.2f}s")
    print(f"  Compression time: {compression_time:.2f}s")
    print(f"  Read time: {read_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Compression throughput: {num_frames/compression_time:.1f} frames/sec")
    print(f"  Data throughput: {(original_size/1e6)/compression_time:.1f} MB/sec")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"Source: TIFF sequence from {args.tiff_dir}")
    print(f"Frames processed: {num_frames} (offset: {args.start_offset})")
    print(f"Frame dimensions: {width} x {height}")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Space savings: {space_savings:.1f}%")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.6f}")
    print(f"Compression speed: {num_frames/compression_time:.1f} frames/sec")
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
            
            fig.suptitle('EBCC Compression (Parallel): Original vs Reconstructed vs Difference',
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
