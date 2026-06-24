# Detector Characterization Toolkit

Tools for characterizing detector performance from illuminated and dark ramp data.

This repository implements a set of analyses used to measure:

* Conversion gain (e⁻/DN)
* Read noise
* Reset (kTC) noise
* Dark current

from FITS ramp data. The code was originally developed for characterization of the SCALES spectrograph detector and has since been generalized to support additional detectors and datasets through YAML-based configuration files.

## Overview

The analysis is organized around datasets rather than specific detectors. A YAML configuration file specifies:

* Data location
* Detector metadata
* Analysis region of interest (ROI)
* Readout modes
* Illuminated ramps
* Dark ramps

The same analysis code can then be applied to multiple detectors or observing runs without modifying the Python source.

## Repository Structure

```text
.
├── gain.py
├── dark_noise.py
├── dark_current.py
├── io.py
├── results/
├── datasets/
    └── various .yaml files
```

### gain.py

Measures detector conversion gain using the two-image difference method.

For each pair of illuminated ramps:

1. Subtract the first read from each ramp.
2. Compute the difference image.
3. Measure signal and variance.
4. Fit the photon transfer curve:

Var(S) = S/g + RN²

where:

* S = signal (DN)
* g = gain (e⁻/DN)
* RN = read noise

Outputs:

* Signal-variance CSV files
* Gain plots
* Summary statistics

### dark_noise.py

Measures:

* Read noise
* Reset (kTC) noise

from dark ramps.

Read noise is derived from consecutive-read differences within a ramp.

Reset noise is derived from differences between the first reads of independently reset ramps.

Outputs:

* Noise summary tables
* Diagnostic correction plots
* Read/reset noise comparison figures

### dark_current.py

Measures dark current from dark ramps by fitting the slope of signal accumulation versus time.

Outputs:

* Dark current estimates
* Ramp-profile plots
* Dark current histograms

### io.py

Utility routines for generating diagnostic difference images and FITS products.

## Configuration

All analyses are configured through YAML files.

Example:

```yaml
dataset:
  name: spectrograph
  head: IFS
  date: "260618"
  datapath: /data/
  outpath: ./results/
  modes:
    - slow52
    - fast10
    - fast06

roi:
  x_start: 833
  x_end: 1020
  y_start: 1284
  y_end: 1500

slow52:
  name: Slow5.2
  frametime: 5.24288

  illuminated:
    ramps: [19776, 19777]

  dark:
    ramp_range: [19867, 19882]
```

## Ramp Selection

Illuminated ramps may be specified in several ways.

### Explicit list

```yaml
illuminated:
  ramps: [19776, 19777, 19778, 19779]
```

### Range

```yaml
illuminated:
  ramp_range: [19776, 19780]
```

### Explicit pairs

```yaml
illuminated:
  pairs:
    - [19776, 19777]
    - [19778, 19779]
```

### Pairing strategies

```yaml
illuminated:
  ramps: [19776, 19777, 19778, 19779]
  pairing: all_pairs
```

Available pairing modes:

* `first_two`
* `adjacent`
* `all_pairs`

This allows characterization results to be compared across multiple ramp combinations.

## Usage

Gain analysis:

```bash
python gain.py datasets/260618_IFS.yaml
```

Read/reset noise:

```bash
python dark_noise.py datasets/260618_IFS.yaml
```

Dark current:

```bash
python dark_current.py datasets/260618_IFS.yaml
```

Diagnostic difference images:

```bash
python io.py datasets/260618_IFS.yaml
```

## Validation Philosophy

The goal of this project is not only to produce detector characterization values but also to evaluate the robustness of those measurements.

Where multiple ramps are available, analyses can be repeated across different ramp pairings to assess:

* Gain stability
* Sensitivity to ramp selection
* Dataset consistency
* Systematic effects

Future development is focused on automated comparison of pairing strategies and uncertainty estimation from ramp-to-ramp variation.

## Dependencies

* Python 3.10+
* NumPy
* SciPy
* Matplotlib
* Astropy
* PyYAML

Install with:

```bash
pip install numpy scipy matplotlib astropy pyyaml
```
