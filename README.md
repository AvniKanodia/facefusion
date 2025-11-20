FaceFusion
==========

> Industry leading face manipulation platform.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
[![Coverage Status](https://img.shields.io/coveralls/facefusion/facefusion.svg)](https://coveralls.io/r/facefusion/facefusion)
![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)


Installation
------------

Be aware, the [installation](https://docs.facefusion.io/installation) needs technical skills and is not recommended for beginners. In case you are not comfortable using a terminal, our [Windows Installer](http://windows-installer.facefusion.io) and [macOS Installer](http://macos-installer.facefusion.io) get you started.


Usage
-----

Run the command:

```
python facefusion.py [commands] [options]

options:
  -h, --help                                      show this help message and exit
  -v, --version                                   show program's version number and exit

commands:
    run                                           run the program
    headless-run                                  run the program in headless mode
    batch-run                                     run the program in batch mode
    force-download                                force automate downloads and exit
    benchmark                                     benchmark the program
    job-list                                      list jobs by status
    job-create                                    create a drafted job
    job-submit                                    submit a drafted job to become a queued job
    job-submit-all                                submit all drafted jobs to become a queued jobs
    job-delete                                    delete a drafted, queued, failed or completed job
    job-delete-all                                delete all drafted, queued, failed and completed jobs
    job-add-step                                  add a step to a drafted job
    job-remix-step                                remix a previous step from a drafted job
    job-insert-step                               insert a step to a drafted job
    job-remove-step                               remove a step from a drafted job
    job-run                                       run a queued job
    job-run-all                                   run all queued jobs
    job-retry                                     retry a failed job
    job-retry-all                                 retry all failed jobs
```


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.


# Benchmark Improvements
-----------------------

### CUDA Benchmarks
```
python facefusion.py benchmark \
    --execution-providers cuda \
    --execution-device-ids 0 \
    --execution-thread-count 8 \
    \
    --processors face_swapper \
    --face-swapper-model hyperswap_1c_256 \
    \
    --face-detector-angles 0 \
    --face-detector-model retinaface \
    --face-detector-size 640x640 \
    --face-detector-score 0.5 \
    --face-landmarker-score 0.5 \
    --face-mask-blur 0.7 \
    --face-mask-padding 20 16 0 16 \
    --face-mask-types box \
    --face-selector-mode many \
    --face-selector-order left-right \
    --reference-face-distance 0.6 \
    \
    --temp-frame-format png \
    --output-video-encoder libx264 \
    --output-video-quality 99 \
    --output-video-preset ultrafast \
    --output-audio-encoder aac \
    --output-image-quality 100
```

Our Version:

| target_path                      | cycle_count | average_run | fastest_run | slowest_run | relative_fps |
| -------------------------------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| .assets/examples/target-240p.mp4 | 5           | 5.45        | 5.40        | 5.51        | 49.54        |

Facefusion 3.5.1:

| target_path                      | cycle_count | average_run | fastest_run | slowest_run | relative_fps |
| -------------------------------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| .assets/examples/target-240p.mp4 | 5           | 5.95        | 5.91        | 5.97        | 45.40        |


Our CUDA changes deliver:
  - 8–9% lower latency (5.95s → 5.45s average)
  - 9% higher FPS (45.4 → 49.54)


### TensorRT Benchmarks
```
python facefusion.py benchmark \
    --execution-providers tensorrt \
    --execution-device-ids 0 \
    --execution-thread-count 8 \
    \
    --processors face_swapper \
    --face-swapper-model hyperswap_1c_256 \
    \
    --face-detector-angles 0 \
    --face-detector-model retinaface \
    --face-detector-size 640x640 \
    --face-detector-score 0.5 \
    --face-landmarker-score 0.5 \
    --face-mask-blur 0.7 \
    --face-mask-padding 20 16 0 16 \
    --face-mask-types box \
    --face-selector-mode many \
    --face-selector-order left-right \
    --reference-face-distance 0.6 \
    \
    --temp-frame-format png \
    --output-video-encoder libx264 \
    --output-video-quality 99 \
    --output-video-preset ultrafast \
    --output-audio-encoder aac \
    --output-image-quality 100
```

Our Version:

| target_path                      | cycle_count | average_run | fastest_run | slowest_run | relative_fps |
| -------------------------------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| .assets/examples/target-240p.mp4 | 5           | 2.88        | 2.84        | 2.89        | 93.86        |


Facefusion 3.5.1:

| target_path                      | cycle_count | average_run | fastest_run | slowest_run | relative_fps |
| -------------------------------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| .assets/examples/target-240p.mp4 | 5           | 3.06        | 3.04        | 3.10        | 88.25        |


Our TensorRT changes deliver:
  - 6% lower latency (3.06s → 2.88s average)
  - 6–7% higher FPS (88.25 → 93.86)


# Summary of Optimizations 
-------------------------

### GPU Acceleration & Performance

  - **Custom TensorRT Wrapper (tensorrt_runner.py)** - Replaces ONNX Runtime's basic TensorRT support with custom engine compilation featuring CuPy zero-copy GPU binding, dynamic power-of-2 batch optimization (up to 64 faces), and SHA1-versioned engine caching that eliminates warmup on subsequent runs

  - **CuPy Zero-Copy Pipeline Integration** - Extensive face_swapper.py modifications eliminate around 50 CPU to GPU transfers per frame by maintaining tensors in GPU memory with cp.asarray direct binding, executing all preprocessing operations (transpose NHWC -> NCHW normalization, clamping) on GPU without NumPy conversion overhead

  - **Custom CUDA Kernels (gpu/kernels.py)** - Runtime-compiled Catmull-Rom bicubic interpolation kernels with CUDA texture objects   (cudaTextureObject_t) deliver faster warping than PyTorch grid_sample while eliminating aliasing through 4x4 sample neighborhoods, processed in 16x16 thread blocks 

  - **Hardware Video Pipeline (gpu_video_pipeline.py)** - Full NVDEC/NVENC pipeline using torchaudio's hardware decode/encode APIs (NOT ffmpe CLI) with multi-stream architecture (separate CUDA streams for decode/compute/encode), faster video I/O through overlapped execution, pinned buffers, and non-blocking transfers with ThreadPoolExecutor pipelining

  - **CV-CUDA Preprocessing (gpu/preprocess.py)** - GPU-native affine warping via cvcuda.warpaffine and NV12 color conversion via cvcuda.cvtcolor bypass CPU entirely when available, with automatic PyTorch grid_sample fallback for compatibility

  - **Optimized Inference Sessions** - Adds SessionOptions configuration with ORT_ENABLE_ALL graph optimization, memory pattern caching, PyTorch CUDA stream binding via cuda_stream provider option for proper synchronization, and automatic serial (GPU) vs parallel (CPU) execution mode selection to avoid thread contention.

### Quality & Temporal Stability

  - **GPU ROI Compositor (gpu/compositor.py)** - 3-level Laplacian pyramid blending in perceptually-correct linear RGB space with sRGB↔Linear conversions, separable Gaussian blur, and spatial dithering eliminates seams, color halos, and banding artifacts that base paste_back operations exhibit, maintaining 8-bit quality without visible posterization.

  - **Hardware Optical Flow Stabilization** - NVIDIA OFA (NvidiaOpticalFlow_2_0) warps previous frames forward and blends with 70% weight when similarity >0.85, reducing flicker through confidence-gated fusion (tau=0.35) 

  - **Adaptive Anomaly Detection** - Tracks quality metrics across 10-frame windows using approximate SSIM; when drops exceed 15% below median, applies 90% previous-frame blending to prevent catastrophic dips while maintaining sharpness during normal operation, addressing quality instability in challenging poses.

  - **Temporal Snap & Seam Reuse** - Eliminates micro-jitter by blending 70% toward previous when SSIM >0.95, and caches 12-pixel seam bands for reuse when face motion <3 pixels, completely removing edge flicker artifacts without per-frame blending cost that base's static composition exhibits.

  - **Face Tracking (face_tracker.py)** - Lucas-Kanade pyramidal optical flow propagates 68 landmarks across frames with One-Euro filtering, reducing expensive detector calls by 5x (every 6 frames vs every frame) while maintaining responsiveness through IoU-based trac matching and confidence gating that skips low-quality detections.

  - **Similarity-Transform Smoothing (temporal_filters.py)** - Decomposes affines into SE(2) components and applies 2nd-order polynomial fitting over 9-frame windows with bounded derivatives (max 3°/frame rotation, 6px translation, 2% scale change), providing physically-plausible smooth transformations that eliminate jitter 

  - **Quantized Color Transfer** - Linear-space gain/bias correction with 1/256 quantization and momentum-based smoothing (0.1 alpha) prevent frame-to-frame color pumping artifacts while maintaining stable appearance, addressing color inconsistency issues in base's direct RGB operations without temporal coherence.

  - **Performance Profiling (profiler.py)** - Thread-safe metrics collection tracks per-stage timing (detector, landmarker, recognizer, swapper) with atomic accumulation and snapshot-reset API for bottleneck identification during optimization, providing visibility 
