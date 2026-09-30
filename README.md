# Bacterial growth curve parameter extraction

Python code for converting OD600 time-series measurements into a table of growth
summaries: maximum observed growth (K), maximum specific growth rate (r), and a
growth-onset indicator (t). It processes multiple curves from an Excel workbook
and exports results and exclusion records for downstream analysis.

**日本語概要:** OD600時系列データの前処理と増殖指標の抽出を自動化する研究用ツールです。Python、NumPy、pandasを用い、各曲線の基準値補正、平滑化、増殖速度・最大増殖量・増殖開始指標の計算を行います。処理結果と除外理由をExcelに出力し、多数の培養条件を比較するためのデータ整備を支援します。

## Research context

This repository supports the study:

> Zhang, S., Sawamura, K. & Ying, B.-W. (2026).
> *Population dynamics of six representative bacteria across hundreds of
> compositionally defined media*. Scientific Data.
> [DOI: 10.1038/s41597-026-07990-x](https://doi.org/10.1038/s41597-026-07990-x)

The study describes 11,686 growth curves from six bacterial species across 664
medium formulations. This repository provides the parameter-extraction script;
the full experimental workbooks are not bundled here. See the paper for the
experimental protocol and data availability information.

## Implementation highlights

- Batch processing: one time column and multiple growth-curve columns in one workbook.
- Preprocessing: numeric parsing, missing-value handling, and per-curve baseline correction.
- Numerical analysis: centered rolling means and finite-difference gradients of log OD600.
- Traceable outputs: baseline values, peak times, review flags, and exclusion reasons.
- Reusable functions: parameter calculation can be imported separately from Excel input/output.

## Repository contents

| File | Purpose |
| --- | --- |
| `calculate_K_r_t.py` | Excel input/output, preprocessing, parameter calculation, and reporting |
| `LICENSE` | MIT license for the code |

## Quick start

The original README specifies Python 3.9.13. The current script was also checked
with Python 3.12.14; the verification environment is listed below.

```bash
git clone https://github.com/shuyangzhang9-JoJo/growth-curve-six-bacteria.git
cd growth-curve-six-bacteria
python -m pip install numpy pandas openpyxl
```

Place an input workbook named `curves.xlsx` in the working directory, then run:

```bash
python calculate_K_r_t.py
```

The script reads the first worksheet, prints processed/excluded curve counts,
and writes `parameter_output.xlsx`.

### Small demonstration without experimental data

Save this snippet as `make_demo_input.py` in the repository directory. It creates
one synthetic growth curve and one flat curve for checking parameter extraction
and exclusion reporting. These are demonstration values, not study data.

```python
import pandas as pd

pd.DataFrame({
    "time_h": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    "growth_01": [0.05, 0.05, 0.05, 0.065, 0.10, 0.18, 0.30, 0.42],
    "no_growth_02": [0.05] * 8,
}).to_excel("curves.xlsx", index=False)
```

```bash
python make_demo_input.py
python calculate_K_r_t.py
```

Expected summary: 2 input curves, 1 processed curve, and 1 excluded curve.
The output contains a `metrics` sheet and an `excluded` sheet.

## Input format

The first column must contain numeric elapsed time in **hours**, in strictly
increasing order. Each remaining column contains the raw OD600 readings of one
sample. Column names become sample identifiers in the output.

| time_h | growth_01 | no_growth_02 |
| ---: | ---: | ---: |
| 0.0 | 0.050 | 0.050 |
| 0.5 | 0.050 | 0.050 |
| 1.0 | 0.050 | 0.050 |
| 1.5 | 0.065 | 0.050 |

Use numeric elapsed hours rather than clock-time strings or absolute timestamps
to keep the reported times relative to the experiment. Missing, non-numeric,
and infinite OD readings are treated as missing during baseline correction.

## Calculation methods

Parameters are calculated directly from the measured time series using the
definitions implemented in `calculate_K_r_t.py`.

| Quantity | Calculation | Interpretation |
| --- | --- | --- |
| Baseline | First valid numeric OD600 reading of each curve | Subtracted from all readings; normally the time-zero value |
| K | Maximum of the centered rolling mean of baseline-corrected OD600 | Maximum observed growth yield in the measurement period |
| r | Maximum of the centered rolling mean of the numerical derivative of natural-log OD600 versus time | Specific growth-rate estimate in h^-1; only positive corrected readings are used |
| t | Time at the start of the earliest window of 5 strictly increasing corrected observations | A growth-onset/lag indicator based on consecutive increases |

The window sizes are configurable. Five strictly increasing observations mean
four positive successive differences. The t calculation uses corrected readings
without rolling-mean smoothing.

## Configuration

Edit the configuration block at the top of `calculate_K_r_t.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `INPUT_XLSX` | `"curves.xlsx"` | Input workbook path, relative to the working directory |
| `OUTPUT_XLSX` | `"parameter_output.xlsx"` | Output workbook path |
| `ROLLING_WINDOW` | `5` | Centered rolling-mean window for K and r, in observations |
| `ALLOW_SINGLE_POINT_K` | `True` | Retain K when a curve has only one positive corrected reading; r is missing |
| `T_CONSECUTIVE_POINTS` | `5` | Number of strictly increasing observations required for t |

NumPy, pandas, and openpyxl are external dependencies. The `re` module used for
sample sorting is part of the Python standard library.

## Outputs and review flags

The `metrics` sheet includes sample identifiers, baseline values, K/r/t values,
the corresponding times, and two flags:

- `K_tail_flag = 1`: the K peak occurs within the final `ROLLING_WINDOW`
  observations. Check whether the experiment covered the growth plateau.
- `r_flag = 2`: the maximum-rate time is at or before 1 hour. Review early readings
  and baseline correction.

The output column `Baseline(time0)` stores the first valid reading, even if the
time-zero reading is missing. For retained single-positive-point curves, r and
the flags are missing.

The `excluded` sheet is created only if curves are excluded. It records sample
names, exclusion reasons, total numeric readings, and positive corrected readings.
Curves without positive corrected readings are excluded; r requires at least two
positive readings. If no increasing window is found, t is missing while the other
available parameters can still be reported.

## Scope and verification

These quantities summarize the observed curves. K depends on measurement duration,
and t is sensitive to small successive increases. The flags support manual review;
they do not establish biological validity or automatically detect contamination.

An end-to-end check using the synthetic example above covered Excel input,
parameter extraction, exclusion reporting, and Excel output. Environment:
Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, and openpyxl 3.1.5. This checks software
execution and the documented file workflow, rather than reproduction of the full
study results.

## Related project and maintainer

For model-based medium selection, see
[Selective_medium_optimization](https://github.com/shuyangzhang9-JoJo/Selective_medium_optimization).

Maintainer: Shuyang Zhang, University of Tsukuba.

## License

The code is distributed under the [MIT License](LICENSE). Consult the data
source for the license applicable to the experimental datasets.

