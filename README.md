[README.md](https://github.com/user-attachments/files/34500345/README.md)
# GIFT Data Package

This package contains implementations and datasets for solving inverse problems using Conditional Diffusion Models (CDM) with different sampling strategies in active learning loops. The methods are applied to the GIFT (Geophysical Imaging and Flow Testing) experimental framework.

## Overview

The GIFT Data Package includes multiple directories, each containing configurations, code, and results for different experimental setups. The primary method utilizes adaptive sampling strategies within active learning loops to efficiently invert for unknown parameters using real and synthetic GIFT data.

## Folder Structure

### `GIFT_adaptive_sampling_synthetic/`

**Purpose:** Adaptive sampling with active learning using synthetic simulated data

Contains all the files needed to run the CDM with adaptive sampling in the active learning loop for solving the inverse problem using **synthetic/simulated data**. This serves as a validation and benchmark against the real data experiments.

**Key Files:**
- `CDM_aps_synthetic.py` - Main script for conditional diffusion model with synthetic data
- `run_GIFT_CDM_aps_synthetic.slurm` - SLURM submission script
- `gift_pflotran_template.in` - PFLOTRAN configuration template
- `gift_pflotran_template_dbase.h5` - PFLOTRAN database template

**Output Directories:**
- `GIFT_CDM_Results/` - Model results and sampled parameters
- `Inverted_field_simulations/` - Inversion results
- `pflotran_input_files/` - Generated input configurations

## Shared Files

Several files appear across multiple directories as they represent common utilities or templates:

- `deception_proc.py` - Post-processing or data handling utility
- `pflotran_tank_sim.py` - PFLOTRAN simulator interface/wrapper
- `soil_compressibility.txt` - Soil parameter data
- `*.srv` files - Survey/configuration files for geophysical surveys

## Results Structure

Each experimental folder contains a `GIFT_CDM_Results/` directory with:
- `sampled_params_stage_X.npy` - Model parameters sampled at stage X
- `field_fit_history.json` - Training/fitting history
- `field_*_true.npy` - True field values for reference/validation

## Usage Notes

- Scripts are designed to run on HPC systems with SLURM job scheduling
- PFLOTRAN simulations are used for forward modeling
- Results are stored as NumPy arrays (.npy) and JSON files
- Each experimental run produces staged results showing convergence through active learning

