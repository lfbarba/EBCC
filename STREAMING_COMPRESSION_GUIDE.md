# Streaming TIFF Compression with EBCC

## Overview

The `compress_tiff_sequence.py` script has been modified to process TIFF sequences in a **streaming fashion**, enabling compression of very large datasets that don't fit entirely in memory.

## Key Features

### 1. **Memory-Efficient Processing**
- Only loads batches of frames into memory at a time
- Configurable batch size based on available RAM and worker count
- Default batch size: `workers × 2` (keeps CPUs busy without excessive memory use)

### 2. **Parallel Compression**
- Uses multiprocessing to compress frames in parallel
- Worker processes share a common EBCC encoder library
- Batch processing ensures workers stay busy while memory stays low

### 3. **Streaming Workflow**

```
For each batch:
  1. Load N frames from disk (N = batch_size)
  2. Normalize frames to [0, 1]
  3. Distribute to worker pool for parallel compression
  4. Write compressed bytes to HDF5 file
  5. Free memory and proceed to next batch
```

## Usage Examples

### Basic Streaming Compression (Sequential)
```bash
python compress_tiff_sequence.py \
  --tiff-dir=/path/to/tiff/files \
  --output compressed.hdf5 \
  --base-cr=200 \
  --residual-target=0.02
```

### Parallel Streaming Compression (Recommended)
```bash
python compress_tiff_sequence.py \
  --parallel \
  --workers 8 \
  --batch-size 16 \
  --tiff-dir=/path/to/tiff/files \
  --output compressed.hdf5 \
  --base-cr=200 \
  --residual-target=0.02
```

### Large Dataset Example (1000+ frames)
```bash
python compress_tiff_sequence.py \
  --parallel \
  --workers 10 \
  --batch-size 20 \
  --max-frames 1251 \
  --tiff-dir=/path/to/large/dataset \
  --output compressed_full.hdf5 \
  --base-cr=200 \
  --residual-target=0.02
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tiff-dir` | str | required | Directory containing TIFF files |
| `--output` | str | `compressed_tiff_sequence.hdf5` | Output HDF5 file path |
| `--start-offset` | int | `150` | Starting frame index |
| `--max-frames` | int | `None` | Maximum frames to process (None = all) |
| `--base-cr` | int | `100` | Base compression ratio (1-300) |
| `--residual-mode` | str | `relative_error_target` | Residual mode: `none`, `relative_error_target`, `max_error_target` |
| `--residual-target` | float | `0.01` | Residual error target |
| `--parallel` | flag | `False` | Enable parallel compression |
| `--workers` | int | `cpu_count-1` | Number of worker processes |
| `--batch-size` | int | `workers×2` | Frames per batch (memory control) |

## Memory Footprint Calculation

**Per-frame memory**: `height × width × 4 bytes` (float32)

**Per-batch memory**: `batch_size × height × width × 4 bytes`

### Examples:
- **1008×1900 frames**, batch_size=16: ~116 MB per batch
- **2016×1400 frames**, batch_size=24: ~271 MB per batch
- **2560×2160 frames**, batch_size=8: ~168 MB per batch

## Performance Considerations

### Choosing Batch Size
- **Small batch** (workers × 1-2): Lower memory, may underutilize CPUs
- **Medium batch** (workers × 2-4): Balanced memory/CPU usage ✅ **Recommended**
- **Large batch** (workers × 8+): Higher memory, better throughput

### Choosing Worker Count
- **Rule of thumb**: Use `cpu_count - 1` to leave one core for system tasks
- **High compression ratios** (base_cr > 150): CPU-bound, use more workers
- **Low compression ratios** (base_cr < 50): I/O-bound, fewer workers may suffice

## Output Format

### Parallel Mode (`--parallel`)
- Dataset: `compressed_vlen` (variable-length uint8 arrays)
- Each frame stored as compressed bytes
- Requires custom decoder (included in script)

### Sequential Mode (default)
- Dataset: `compressed` (float32 with EBCC HDF5 filter)
- Standard HDF5 filter interface
- Readable by any HDF5 tool with EBCC filter installed

## Limitations

### SPIHT Dimension Limit
The residual encoder has a maximum dimension of **2047 pixels** per axis.

**Workarounds for larger frames**:
1. **Disable residual compression**: Use `--residual-mode=none`
2. **Tile large frames**: Split into smaller chunks (requires code modification)
3. **Downsample**: Reduce frame size before compression (lossy)

### Example for 2560×2160 frames:
```bash
python compress_tiff_sequence.py \
  --parallel \
  --workers 8 \
  --residual-mode none \
  --base-cr 50 \
  --tiff-dir=/path/to/large/frames \
  --output compressed_jp2only.hdf5
```

## Verification and Quality Metrics

The script automatically:
1. Decompresses all frames for verification
2. Computes quality metrics (PSNR, SSIM) in streaming mode
3. Generates comparison visualizations for selected frames

**Note**: Decompression is currently sequential but doesn't load all frames into memory simultaneously.

## Comparison: Streaming vs. Original

| Aspect | Original | Streaming |
|--------|----------|-----------|
| Memory usage | All frames loaded | Only batch loaded |
| 300 frames (1008×1900) | ~3.4 GB | ~116 MB (batch=16) |
| 1251 frames (2016×1400) | ~14.1 GB | ~271 MB (batch=24) |
| Processing mode | Sequential or parallel | Sequential or parallel |
| Performance | Same | Same (with good batch size) |

## Example Run

```bash
$ python compress_tiff_sequence.py --parallel --workers 8 --batch-size 16 \
    --max-frames 50 --tiff-dir=./file_9_extracted --output test.hdf5

======================================================================
EBCC TIFF Sequence Compression (Streaming Mode)
======================================================================

1. Scanning TIFF sequence...
Found 1251 TIFF files
Processing 50 frames (offset: 150)
Image dimensions: 2016 x 1400
Scanning 50 frames to determine value range...
Value range: [64.0, 1629.0]

2. Configuring EBCC compression...
Residual mode: relative_error_target
Residual target: 0.02
Base compression ratio: 200

3. Compressing data with EBCC (streaming mode)...
Parallel mode enabled: using 8 workers
Batch size: 16 frames (keeps 8 workers busy)
Memory footprint per batch: ~180.63 MB
Processing 50 frames in 4 batches...
Compressing frames: 100%|██████████| 50/50 [01:50<00:00, 2.20s/frame]

4. Analyzing compression performance...
Compression ratio (vs uint16): 8.84:1
Space savings (vs uint16): 88.7%

5. Computing quality metrics (streaming)...
Average PSNR: 49.16 dB
Average SSIM: 0.996035

✅ Done!
```

## Tips for Very Large Datasets

1. **Test with small subset first**: Use `--max-frames 100` to validate parameters
2. **Monitor memory usage**: Check actual RAM consumption with Activity Monitor/htop
3. **Adjust batch size**: If memory issues occur, reduce `--batch-size`
4. **Use background processing**: Run with `nohup` or `tmux` for long-running jobs
5. **Check intermediate results**: Script saves to HDF5 incrementally

## Implementation Details

The streaming implementation uses three key strategies:

1. **Metadata Scanning**: Samples subset of frames to determine value range without loading all data
2. **Batch Loading**: `load_single_tiff_frame()` loads frames on-demand
3. **Chunked HDF5**: Uses `chunks=(1, height, width)` for per-frame compression

This approach is inspired by the HEVC streaming encoder in `stream_gray10.py`, adapted for EBCC compression.
