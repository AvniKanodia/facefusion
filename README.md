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


Optimizations 
-------------

## Performance & GPU Acceleration

- Added GPU video pipeline with **NVDEC hardware decoding** and **NVENC hardware encoding**
- Implemented **TensorRT runner** for accelerated model inference
- Added **CUDA-based GPU compositor** with custom blending kernels
- Introduced GPU-accelerated preprocessing modules for frame operations
- Enabled **end-to-end GPU processing**, keeping frames entirely in CUDA memory

## Architecture Improvements

- Consolidated processor modules into unified single-file implementations:  
  `age_modifier`, `deep_swapper`, `expression_restorer`,  
  `face_debugger`, `face_editor`, `face_enhancer`, `face_swapper`,  
  `frame_colorizer`, `frame_enhancer`, `lip_syncer`
- Replaced the previous translator/locals/sanitizer modules with a unified **wording module**
- Added a lightweight **face tracker** to reduce repeated per-frame detections
- Added a new **profiler module** for real-time performance monitoring and debugging
