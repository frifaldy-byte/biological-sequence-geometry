# Biological Sequence Geometry

**Mathematical Modeling of Dose-Matched Biological Stimulus Sequences**

## Overview

This repository accompanies a mathematical study of how the order and temporal arrangement of biological stimuli affect accumulated responses when total exposure to each stimulus is held constant.

The study considers a two-stimulus system in which different delivery sequences can produce different final responses despite having identical cumulative doses. The analysis is formulated in cumulative-exposure space, where stimulus trajectories are examined through path integrals and a weighted-area representation.

The mathematical treatment includes sequence reversal, local Lie-bracket analysis, and higher-order path-dependent quantities. Numerical computations are used to examine the resulting relationships and assess their behavior under different parameter configurations.

## Mathematical Framework

The model is formulated as a nonlinear ordinary differential equation system with two interacting stimulus inputs and an accumulated response variable.

The analysis focuses on the following components:

- **Two-stimulus response model:** formulation of the dynamics governing stimulus-dependent accumulation.
- **Cumulative-exposure coordinates:** representation of stimulus delivery in terms of accumulated doses.
- **Weighted-area formulation:** characterization of path-dependent response differences using Green's theorem.
- **Sequence reversal:** analytical comparison of protocols with reversed stimulus order.
- **Lie-bracket analysis:** examination of local order effects through the associated vector-field structure.
- **Higher-order path coordinates:** construction of additional quantities for distinguishing stimulus trajectories beyond lower-order path information.

These components provide a mathematical description of sequence-dependent behavior within the proposed model.

## Computational Analysis

The computational work is implemented in Python and supports the analytical and numerical investigations presented in the accompanying study.

The analyses include:

- Numerical evaluation of dose-matched stimulus protocols.
- Comparison of sequential, simultaneous, and alternating stimulus delivery.
- Evaluation of weighted-area effects across cumulative-exposure space.
- Identification of sequence-reversal boundaries.
- Numerical assessment of local Lie-bracket approximations.
- Comparison of higher-order path-dependent quantities.
- Local parameter sensitivity analysis.
- Global sensitivity analysis and parameter perturbation experiments.

Numerical integration and computational checks are used to examine the model equations and compare analytical expressions with their numerical counterparts.

## Repository Contents

The repository contains the computational materials associated with the study:

| File | Description |
|---|---|
| `PaperBioLieAlgebra.ipynb` | Jupyter notebook containing mathematical and computational analyses. |
| `Phase_1.py` | Python implementation for the first analysis phase. |
| `Phase_2.py` | Python implementation for the second analysis phase. |
| `Phase_3.py` | Python implementation for the third analysis phase. |
| `Phase_4B.py` | Python implementation for the additional analysis phase. |
| `Biological_Sequence_Geometry_Final_Package` | Packaged research materials associated with the final computational work. |
| `LICENSE` | MIT License. |

The Python scripts and notebook provide the computational components used throughout the investigation. The packaged materials are included as a consolidated research resource.

## Requirements

The computational implementation uses Python 3 and the following libraries:

- NumPy
- SciPy
- Matplotlib

Install the dependencies using:

```bash
pip install numpy scipy matplotlib
```

For notebook-based analysis, Jupyter Notebook or JupyterLab is also required.

## Scope and Limitations

This study presents a theoretical mathematical framework for examining stimulus-order dependence. The parameter values used in the numerical experiments are illustrative and are not estimated from or fitted to experimental cellular data.

The findings therefore describe the mathematical behavior of the proposed system under its specified assumptions. They should not be interpreted as direct experimental evidence of a particular cellular mechanism or as clinical predictions.

Experimental validation and parameter calibration using suitable biological data would be necessary to establish the model's applicability to specific biological systems.

## License

This repository is distributed under the [MIT License](LICENSE).
