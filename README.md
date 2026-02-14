# Simultaneous Estimation and Variable Selection for Interval-Censored Competing Risk Data

**Author:** Lou Yichen (The Chinese University of Hong Kong)

## Overview

This repository contains the MATLAB code accompanying the paper *"Simultaneous Estimation and Variable Selection for Interval-Censored Competing Risk Data"*.

## Repository Structure

```
├── main_AL.m        # Adaptive LASSO (ALASSO) implementation
├── main_PL.m        # Other penalized likelihood methods
├── Qpenalty.m       # Penalty function selection
└── FRAM/            # Framingham Heart Study application
    ├── main_AL.m    # ALASSO for Framingham data
    └── main_PL.m    # Other penalties for Framingham data
```

## Usage

### Simulation Studies

- **ALASSO:** Run `main_AL.m`
- **Other penalties:** Run `main_PL.m`

### Changing the Penalty Function

1. Select the desired penalty in `Qpenalty.m`.
2. Update the corresponding candidate tuning parameters (`lambset`) in the main function.

### Framingham Heart Study Application

The application code is located in the `FRAM/` directory.

1. Download the Framingham Heart Study dataset as a CSV file.
2. Place the CSV file in the `FRAM/` directory.
3. Run `main_AL.m` or `main_PL.m` as needed.

> **Note:** The original Framingham data is not included in this repository due to potential copyright restrictions.

## Requirements

- MATLAB

## Contact

For any questions regarding usage, please contact the corresponding author.
