# ThermoTrace: Physics-Informed U-Net Framework for Non-Invasive Inverse Heat Source Mapping

## Overview

ThermoTrace is a Scientific Machine Learning (SciML) project that solves the inverse heat transfer problem using deep learning. The system reconstructs internal heat source distributions from surface temperature measurements, enabling non-invasive heat source localization and intensity estimation.

The project combines Finite Difference Method (FDM) simulation, synthetic data generation, Physics-Informed Neural Networks (PINNs), and a U-Net architecture to learn the mapping between temperature fields and hidden heat sources.

---

## Key Features

* Custom FDM solver for 2D steady-state heat conduction
* Synthetic dataset generation pipeline
* Physics-informed U-Net architecture
* Adaptive physics-based loss weighting
* Multi-source heat source localization
* Noise-robust inverse heat source reconstruction
* Real-time inference after training

---

## Problem Statement

Given a 64×64 surface temperature field measured on a steel plate, predict:

* Location of internal heat sources
* Spatial distribution of heat generation
* Heat source intensity

This represents a classical inverse heat transfer problem, which is mathematically ill-posed and challenging to solve using traditional optimization methods.

---

## Dataset

A custom synthetic dataset was generated using a physics-based FDM solver.

### Dataset Statistics

| Split      | Samples |
| ---------- | ------- |
| Training   | 8,000   |
| Validation | 1,000   |
| Test       | 1,000   |
| Total      | 10,000  |

### Data Generation Process

1. Randomly generate 1–3 circular heat sources.
2. Solve the Poisson heat conduction equation using FDM.
3. Generate corresponding temperature fields.
4. Add Gaussian sensor noise.
5. Store temperature fields and source maps as training pairs.

Dataset size: ~100 MB

---

## Model Architecture

The project uses a Physics-Informed U-Net Encoder-Decoder architecture.

### Input

* 64×64 normalized temperature field

### Output

* 64×64 heat source intensity map

### Architecture

* Encoder with 4 downsampling stages
* Bottleneck layer
* Decoder with skip connections
* ReLU-constrained non-negative output

Total trainable parameters: ~7.77 Million

---

## Technologies Used

* Python
* PyTorch
* NumPy
* SciPy
* Matplotlib
* Jupyter Notebook
* Google Colab
* NVIDIA T4 GPU

---

## Performance

| Metric                            | Result                     |
| --------------------------------- | -------------------------- |
| Source Detection Accuracy         | 87.90%                     |
| Median Localization Error         | 0.59 Grid Cells (~0.93 mm) |
| Median Intensity Estimation Error | 18.90%                     |
| Normalized Test MSE               | 0.000391                   |

---

## Repository Structure

```text
ThermoTrace/
│
├── data/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
│
├── dataset.py
├── model.py
├── train.py
├── verify.ipynb
│
└── README.md
```

---

## Applications

* Building Energy Diagnostics
* HVAC Fault Detection
* Thermal Condition Monitoring
* Industrial Equipment Inspection
* Data Center Thermal Management
* Non-Destructive Thermal Analysis

---

## Future Work

* 3D Heat Transfer Modeling
* Real Experimental Sensor Data
* Transient Heat Conduction
* Convective Boundary Conditions
* Neural Operator-Based Approaches
* Edge Deployment for Real-Time Monitoring

---

## Author

Anuj Sudhir Pathak

Mechanical Engineering

Dr. D. Y. Patil Institute of Technology

Pune, India
