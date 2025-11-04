# EBCC Compression for TIFF Sequences - Setup Complete ✅

## Environment Setup

A Python virtual environment has been created and configured at:
- **Path**: `/Users/lfbarba/GitHub/EBCC/.venv`
- **Python Version**: 3.13.4
- **Status**: ✅ Active and configured

## Installed Packages

The EBCC package has been successfully installed in editable mode with all development dependencies:
- `ebcc` - Error Bounded Climate Compressor (core package)
- `numpy`, `h5py`, `xarray` - Core scientific computing libraries
- `zarr`, `zarr-any-numcodecs` - Zarr support
- `pytest`, `pytest-benchmark` - Testing framework
- `pandas`, `matplotlib`, `tqdm` - Data analysis and visualization
- `PIL/Pillow` - Image processing (for TIFF loading)
- And many more dependencies for comprehensive functionality

## Compression Script

A new script has been created: `compress_tiff_sequence.py`

This script:
1. Loads TIFF files from a directory
2. Applies EBCC compression using HDF5 filter
3. Computes quality metrics (PSNR, SSIM)
4. Generates comparison visualizations

### Usage Examples

**Basic usage with default settings:**
```bash
python compress_tiff_sequence.py
```

**Custom TIFF directory and parameters:**
```bash
python compress_tiff_sequence.py \
  --tiff-dir /path/to/tiff/files \
  --output my_compressed_data.hdf5 \
  --start-offset 0 \
  --max-frames 500 \
  --base-cr 100 \
  --residual-mode relative_error_target \
  --residual-target 0.01
```

**Different compression settings:**
```bash
# Higher compression (lower quality)
python compress_tiff_sequence.py --base-cr 200 --residual-target 0.05

# Lower compression (higher quality)
python compress_tiff_sequence.py --base-cr 50 --residual-target 0.005

# No residual compression (base only)
python compress_tiff_sequence.py --base-cr 100 --residual-mode none
```

### Command-Line Arguments

- `--tiff-dir`: Directory containing TIFF files (default: `/Users/lfbarba/GitHub/sdate/data/ct_files/file_3_extracted`)
- `--output`: Output HDF5 file path (default: `compressed_tiff_sequence.hdf5`)
- `--start-offset`: Starting frame index (default: 150)
- `--max-frames`: Maximum number of frames to process (default: 300)
- `--base-cr`: Base compression ratio, higher = more compression (default: 100)
- `--residual-mode`: Residual compression mode
  - `none`: Base compression only
  - `relative_error_target`: Target relative error (recommended)
  - `max_error_target`: Target absolute max error
- `--residual-target`: Error target value (default: 0.01)
- `--compare-frames`: Frame indices to visualize (default: 0, 75, 150, 299)

## Test Run Results

Successfully compressed 300 TIFF frames with the following results:

### Compression Performance
- **Original Size**: 2298.24 MB (32-bit float)
- **Compressed Size**: 223.02 MB (213 MB on disk)
- **Compression Ratio**: 10.30:1
- **Space Savings**: 90.3%
- **Bits per Pixel**: 32.00 → 3.11

### Quality Metrics
- **PSNR**: 56.80 dB (Overall), 55.79 dB (Average per frame)
- **SSIM**: 0.999899
- **Max Relative Error**: 0.009658 (well within target of 0.01)
- **Max Absolute Error**: 25.5 (in original value range)

### Data Details
- **Source**: TIFF sequence from `/Users/lfbarba/GitHub/sdate/data/ct_files/file_3_extracted`
- **Frames**: 300 (frames 150-449)
- **Dimensions**: 1008 × 1900 pixels per frame
- **Original Data Type**: uint16
- **Original Value Range**: [288, 2928]

### Output Files
- `compressed_tiff_sequence.hdf5` - Compressed HDF5 file (213 MB)
- `compressed_tiff_sequence_comparison.png` - Visual comparison of original vs compressed

## How EBCC Compression Works

EBCC (Error Bounded Climate Compressor) uses a two-layer approach:

1. **Base Layer (JPEG2000)**:
   - Lossy compression using JPEG2000 codec
   - Compression ratio controlled by `--base-cr` parameter
   - Higher values = more compression but lower quality

2. **Residual Layer (Optional)**:
   - Compresses the error between original and base-compressed data
   - Uses wavelet encoding and sparsification
   - Ensures error stays within specified bounds
   - Modes:
     - `relative_error_target`: Error relative to data range (recommended)
     - `max_error_target`: Absolute maximum error
     - `none`: No residual (base compression only)

## Reading Compressed Data

To read the compressed data back:

```python
import h5py
import os
from ebcc import EBCC_FILTER_DIR

# Set HDF5 plugin path before importing h5py
os.environ["HDF5_PLUGIN_PATH"] = EBCC_FILTER_DIR

# Open and read
with h5py.File('compressed_tiff_sequence.hdf5', 'r') as f:
    # Read metadata
    print(f"Original range: [{f.attrs['original_min']}, {f.attrs['original_max']}]")
    print(f"Frames: {f.attrs['num_frames']}")
    
    # Read compressed data (automatically decompressed)
    data = f['compressed'][:]
    
    # Denormalize back to original range
    min_val = f.attrs['original_min']
    max_val = f.attrs['original_max']
    data_original = data * (max_val - min_val) + min_val
```

## Comparison with HEVC Video Compression

From the notebook (`TiffProjectionHVEC.ipynb`), the same data was compressed using HEVC 10-bit:
- HEVC uses temporal compression (inter-frame prediction)
- EBCC compresses each frame independently
- EBCC provides error bounds guarantees
- HEVC is optimized for visual quality
- EBCC is optimized for scientific data accuracy

## Next Steps

1. **Experiment with different compression settings**:
   - Try different `--base-cr` values (50-200)
   - Adjust `--residual-target` based on acceptable error
   - Compare `relative_error_target` vs `max_error_target` modes

2. **Process other TIFF sequences**:
   - Available: file_1 through file_12 in ct_files directory
   - Adjust `--tiff-dir` parameter

3. **Integrate with workflows**:
   - The compressed HDF5 files can be read by any HDF5-compatible tool
   - CDO (Climate Data Operators) integration available
   - NetCDF support through h5netcdf

4. **Batch processing**:
   - Create scripts to process multiple sequences
   - Compare compression performance across datasets

## Troubleshooting

### Filter not found error
If you get "Unknown compression filter number: 308":
- Ensure `HDF5_PLUGIN_PATH` is set before importing h5py
- Check that libh5z_ebcc.dylib exists in `src/build/lib/`

### Import errors
If you get import errors:
- Activate the virtual environment
- Reinstall: `pip install -e ".[dev]"`

### Build errors
If the C extension doesn't build:
- Ensure git submodules are initialized: `git submodule update --init --recursive`
- Install CMake: `brew install cmake`
- Check compiler is available: `which clang`

## References

- EBCC Paper: [arXiv:2510.22265](https://arxiv.org/abs/2510.22265)
- GitHub Repository: [spcl/EBCC](https://github.com/spcl/EBCC)
- JPEG2000: OpenJPEG library
- Compression backend: ZSTD for residual layer
