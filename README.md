# growth-curve-six-bacteria

This repository contains the code used to estimate growth parameters (K, rmax, and t)
from OD600 time-series data in:

[Population Dynamics of Six Representative Bacteria across Hundreds of Compositionally Defined Media]

## Requirements

Python 3.9.13

Required packages:
- numpy
- pandas
- re

## Description

The script fits growth curves using a logistic model and extracts:
- Carrying capacity (K)
- Maximum growth rate (rmax)
- log time (t)

Baseline correction is performed using the time=0 OD600 value.
