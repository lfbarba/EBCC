#!/usr/bin/env python3
"""
Read and verify EBCC compressed TIFF sequence data

This script demonstrates how to read back EBCC compressed data
and perform basic verification and analysis.
"""

import os
import sys

# Set up HDF5 plugin path - MUST be done before importing h5py
from ebcc import EBCC_FILTER_DIR
os.environ["HDF5_PLUGIN_PATH"] = EBCC_FILTER_DIR

import h5py
import numpy as np
import argparse
import ctypes
from ctypes import c_size_t, c_float, c_uint8, c_void_p, POINTER


def main():
    parser = argparse.ArgumentParser(
        description='Read and verify EBCC compressed TIFF sequence'
    )
    parser.add_argument(
        '--input',
        default='compressed_tiff_sequence.hdf5',
        help='Input HDF5 file with EBCC compressed data'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=None,
        help='Display specific frame (default: show summary only)'
    )
    parser.add_argument(
        '--export-frame',
        type=int,
        default=None,
        help='Export specific frame as PNG'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EBCC Compressed Data Reader")
    print("=" * 70)
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    print(f"\n📂 Opening: {args.input}")
    print(f"   File size: {os.path.getsize(args.input) / 1e6:.2f} MB")
    
    with h5py.File(args.input, 'r') as f:
        # Display metadata
        print("\n📋 Metadata:")
        print(f"   Original value range: [{f.attrs['original_min']}, {f.attrs['original_max']}]")
        print(f"   Number of frames: {f.attrs['num_frames']}")
        print(f"   Frame dimensions: {f.attrs['width']} x {f.attrs['height']}")
        print(f"   Base compression ratio: {f.attrs['base_cr']}")
        print(f"   Residual mode: {f.attrs['residual_mode']}")
        if 'residual_target' in f.attrs:
            print(f"   Residual target: {f.attrs['residual_target']}")
        
        # Access compressed dataset
        if 'compressed' in f:
            dset = f['compressed']
            print(f"\n📊 Dataset Info:")
            print(f"   Shape: {dset.shape}")
            print(f"   Dtype: {dset.dtype}")
            print(f"   Chunks: {dset.chunks}")
            print(f"   Compression: {dset.compression}")
            
            # Read normalized data
            print(f"\n🔄 Reading compressed data (HDF5 filter)...")
            data_normalized = dset[:]
            print(f"   Data range (normalized): [{data_normalized.min():.6f}, {data_normalized.max():.6f}]")
            
            # Denormalize to original range
            min_val = f.attrs['original_min']
            max_val = f.attrs['original_max']
            
            if max_val > min_val:
                data_original = data_normalized * (max_val - min_val) + min_val
            else:
                data_original = data_normalized.copy()
        elif 'compressed_vlen' in f:
            # vlen dataset created by parallel mode: stored as per-frame compressed bytes
            dset = f['compressed_vlen']
            print(f"\n📊 Dataset Info (vlen compressed frames):")
            print(f"   Shape: {dset.shape}")
            print(f"   Dtype: {dset.dtype}")
            print("   Stored as variable-length uint8 compressed frames")
            
            # Need to decode each frame using native decoder
            print(f"\n🔄 Decoding compressed vlen frames using native decoder...")
            # Load shared library and setup ctypes
            from ebcc import EBCC_FILTER_PATH
            lib = ctypes.CDLL(EBCC_FILTER_PATH)
            lib.ebcc_decode.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
            lib.ebcc_decode.restype = ctypes.c_size_t
            lib.free_buffer.argtypes = (ctypes.c_void_p,)
            lib.free_buffer.restype = None

            num_frames = f.attrs['num_frames']
            height = f.attrs['height']
            width = f.attrs['width']
            data_normalized = np.zeros((num_frames, height, width), dtype=np.float32)

            for i in range(num_frames):
                comp = np.ascontiguousarray(dset[i])
                comp_ptr = comp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                out_ptr = ctypes.POINTER(ctypes.c_float)()
                out_size = lib.ebcc_decode(comp_ptr, comp.size, ctypes.byref(out_ptr))
                buf = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float * out_size)).contents
                arr = np.frombuffer(buf, dtype=np.float32).copy()
                lib.free_buffer(out_ptr)
                data_normalized[i] = arr.reshape((height, width))

            min_val = f.attrs['original_min']
            max_val = f.attrs['original_max']
            if max_val > min_val:
                data_original = data_normalized * (max_val - min_val) + min_val
            else:
                data_original = data_normalized.copy()
        else:
            raise RuntimeError("No recognized compressed dataset found ('compressed' or 'compressed_vlen')")
        
        print(f"   Data range (denormalized): [{data_original.min():.2f}, {data_original.max():.2f}]")
        
        # Statistics
        print(f"\n📈 Statistics (denormalized):")
        print(f"   Mean: {data_original.mean():.2f}")
        print(f"   Std: {data_original.std():.2f}")
        print(f"   Median: {np.median(data_original):.2f}")
        
        # Display specific frame if requested
        if args.frame is not None:
            num_frames = f.attrs['num_frames']
            if 0 <= args.frame < num_frames:
                print(f"\n🖼️  Frame {args.frame}:")
                frame = data_original[args.frame]
                print(f"   Range: [{frame.min():.2f}, {frame.max():.2f}]")
                print(f"   Mean: {frame.mean():.2f}")
                print(f"   Std: {frame.std():.2f}")
            else:
                print(f"\n⚠️  Frame {args.frame} out of range (0-{num_frames-1})")
        
        # Export frame as PNG if requested
        if args.export_frame is not None:
            num_frames = f.attrs['num_frames']
            if 0 <= args.export_frame < num_frames:
                try:
                    from PIL import Image
                    
                    frame = data_original[args.export_frame]
                    
                    # Normalize to 0-255 for PNG
                    frame_normalized = (frame - frame.min()) / (frame.max() - frame.min())
                    frame_uint8 = (frame_normalized * 255).astype(np.uint8)
                    
                    output_path = f"frame_{args.export_frame}.png"
                    img = Image.fromarray(frame_uint8, mode='L')
                    img.save(output_path)
                    
                    print(f"\n💾 Exported frame {args.export_frame} to: {output_path}")
                    
                except ImportError:
                    print("\n⚠️  PIL/Pillow not available - cannot export PNG")
                except Exception as e:
                    print(f"\n❌ Error exporting frame: {e}")
            else:
                print(f"\n⚠️  Frame {args.export_frame} out of range (0-{num_frames-1})")
    
    print("\n✅ Done!")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
