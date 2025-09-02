# Sobel Edge Detection – Parallel Computing Project

This project implements **Sobel Edge Detection** for grayscale images using **sequential**, **OpenMP**, and **MPI** approaches. It was developed as part of a **Parallel Computing class** to demonstrate performance improvements through parallelization.

---

## Table of Contents

* [Project Description](#project-description)
* [Features](#features)
* [Dependencies](#dependencies)
* [Build and Run Instructions](#build-and-run-instructions)

---

## Project Description

Sobel edge detection is a widely-used technique in image processing to highlight edges. This project includes three implementations:

1. **Sequential** – Standard Sobel operator using OpenCV.
2. **OpenMP Parallel** – Parallelizes the computation across CPU threads.
3. **MPI Parallel** – Distributes the image processing across multiple processes.

The project measures execution time for each approach and outputs the edge-detected image.

---

## Features

* Sequential Sobel edge detection (`sobel_edge_detection.cpp`)
* Multi-threaded edge detection using OpenMP (`sobel_edge_detection_omp.cpp`)
* Multi-process edge detection using MPI (`sobel_edge_detection_mpi.cpp`)
* Script to upscale images using OpenCV (`upscale.py`)
* Easy build and run with **CMake** and provided `run.sh` script

---

## Dependencies

* **C++17 compiler** (e.g., GCC, Clang)
* **OpenCV** (core, imgproc, highgui)
* **OpenMP**
* **MPI** (e.g., MPICH or OpenMPI)
* **Python 3** (for image upscaling)
* **CMake** (≥ 3.1)

---

## Build and Run Instructions

Build the project using CMake:

```bash
mkdir build
cd build
cmake ..
make
```

Run all algorithms easily using the provided script:

```bash
./run.sh
```

This script executes:

1. **Sequential Sobel Edge Detection**
2. **OpenMP Sobel Edge Detection** (with 8 threads)
3. **MPI Sobel Edge Detection** (with 3 processes)

The edge-detected images will be saved in the same folder as the input image with suffixes:

* `_edge_seq.jpg`
* `_edge_omp.jpg`
* `_edge_mpi.jpg`
