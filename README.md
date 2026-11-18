# Biological Sequence Geometry

**Mathematical Modeling of Dose-Matched Biological Stimulus Sequences**

## Overview

This project develops a mathematical framework for examining how the order and temporal arrangement of biological stimuli influence accumulated cellular responses, even when total exposure doses are identical.

The framework describes stimulus sequences in cumulative-exposure coordinates and uses weighted-area integrals to characterize differences between dose-matched protocols. Lie-bracket expansions and higher-order path-dependent state variables are also examined.

## Mathematical Framework

The main components include:

- A nonlinear two-stimulus response model.
- Weighted-area characterization using Green's theorem.
- Analytical conditions for sequence reversal.
- Local Lie-bracket analysis.
- Higher-order path-dependent state construction.
- Parameter sensitivity and robustness analysis.

## Computational Analysis

Python is used for numerical integration, mathematical verification, parameter analysis, and figure generation.

The computational work includes:

- Dose-matched protocol comparisons.
- Weighted sequence-sensitivity landscapes.
- Sequence-reversal boundary analysis.
- Lie-bracket approximation checks.
- Higher-dimensional path comparisons.
- Local and global sensitivity analyses.

## Requirements

The main computational dependencies are:

- Python 3
- NumPy
- SciPy
- Matplotlib

Install the packages using:

```bash
pip install numpy scipy matplotlib
```

## Repository Contents

This repository contains the Python implementation, numerical analyses, and graphical outputs associated with the mathematical framework. Figures and computational results are organized by analysis phase.

## Scope and Limitations

The model is a theoretical mathematical framework. Its parameter values are used for computational investigation and are not fitted to experimental cellular data. The results should therefore be interpreted as mathematical findings within the specified model, rather than empirical evidence of a particular biological mechanism.

## License

License information will be added separately.
