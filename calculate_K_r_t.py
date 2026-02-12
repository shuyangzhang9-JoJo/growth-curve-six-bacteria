# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# ========== CONFIGURATION ==========
INPUT_XLSX = r"curves.xlsx"
OUTPUT_XLSX = "parameter_output.xlsx"

ROLLING_WINDOW = 5
ALLOW_SINGLE_POINT_K = True

T_CONSECUTIVE_POINTS = 5 
# ====================================


def ensure_time_index(df, time_col=None):
    """Ensure the first column is used as time index."""
    if time_col is None:
        time_col = df.columns[0]
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found.")

    df = df.dropna(subset=[time_col]).copy()
    df.set_index(time_col, inplace=True)
    return df


def time_index_to_numeric_hours(index):
    """Convert index values to numeric hours."""
    # Try direct numeric conversion
    try:
        return pd.to_numeric(index, errors='raise').astype(float)
    except Exception:
        pass

    # Try datetime conversion
    try:
        dt = pd.to_datetime(index)
        ns = dt.view('int64')
        return ns / 1e9 / 3600.0
    except Exception:
        pass

    # Try best-effort numeric fallback
    arr = pd.to_numeric(index, errors='coerce')
    if np.isnan(arr).all():
        raise ValueError(f"Unable to interpret time index, sample values: {list(index[:5])}")
    return arr.astype(float)


def smooth_series(series, window=5):
    """Apply centered rolling mean smoothing."""
    return series.rolling(window, center=True, min_periods=1).mean()


def baseline_correct(series: pd.Series) -> tuple[pd.Series, float]:
    """
    Baseline correction using the first valid reading in this curve as blank (typically time=0).
    Returns (corrected_series, baseline_value).
    """
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    s_valid = s.dropna()
    if s_valid.empty:
        return s, np.nan
    baseline = float(s_valid.iloc[0])
    return s - baseline, baseline


def compute_t_lagtime(corrected_series: pd.Series, consecutive_points: int = 5):
    """
    t (lag time) definition:
    earliest time where 'consecutive_points' values are strictly increasing.
    Return (t_index, t_hours). If not found: (NaN, NaN).
    """
    s = corrected_series.dropna()
    if len(s) < consecutive_points:
        return np.nan, np.nan

    x_hours = time_index_to_numeric_hours(s.index)
    vals = s.values.astype(float)

    for i in range(0, len(vals) - consecutive_points + 1):
        window = vals[i:i + consecutive_points]
        if np.all(np.diff(window) > 0):  # strict increase
            return s.index[i], float(x_hours[i])

    return np.nan, np.nan


def getK(series, window=5):
    """Compute K using smoothed OD (baseline-corrected)."""
    smoothed = smooth_series(series, window=window)
    max_idx = smoothed.idxmax()
    return max_idx, float(smoothed.max())


def getr(series, window=5):
    """
    Compute r based on max slope of log(OD) vs time (1/h).
    Requires OD > 0.
    """
    x_hours = time_index_to_numeric_hours(series.index)
    y_log = np.log(series.values)

    gradients = np.gradient(y_log, x_hours)
    smoothed_grad = smooth_series(pd.Series(gradients, index=series.index), window)

    max_idx = smoothed_grad.idxmax()
    return max_idx, float(smoothed_grad.max())


def calculate_growth_parameters(df, rolling_window=5, allow_single_point_k=False, t_points=5):
    """
    Compute K, r, and t for each growth curve (each column).
    Applies baseline correction using first valid reading as blank.

    Notes:
    - r uses log(OD), so only positive (baseline-corrected) values are used.
    """
    results = []
    excluded = []

    for col in df.columns:
        raw = df[col]

        # Baseline correction
        corrected, baseline = baseline_correct(raw)

        total_points = int(pd.to_numeric(raw, errors='coerce').dropna().shape[0])

        if corrected.dropna().empty:
            excluded.append({
                "Sample": col,
                "Reason": "All values are NaN after parsing",
                "TotalPoints": total_points,
                "PositivePoints": 0
            })
            continue

        # t (lag time) based on corrected curve
        t_idx, t_h = compute_t_lagtime(corrected, consecutive_points=t_points)

        corrected_nonan = corrected.dropna()

        # r requires OD>0
        corrected_pos = corrected_nonan[corrected_nonan > 0]
        pos_points = int(corrected_pos.shape[0])

        # If no positive points -> cannot compute r
        if pos_points == 0:
            excluded.append({
                "Sample": col,
                "Reason": "No positive values after baseline correction (cannot compute log-based r)",
                "TotalPoints": total_points,
                "PositivePoints": pos_points
            })
            continue

        try:
            if pos_points >= 2:
                # K on corrected curve (not restricted to >0)
                k_time, k_val = getK(corrected_nonan, window=rolling_window)

                # Flag: K occurs at tail end
                idx_list = list(corrected_nonan.index)
                k_pos = idx_list.index(k_time)
                K_tail_flag = 1 if k_pos >= len(idx_list) - rolling_window else 0

                # r on positive corrected only
                r_time, r_val = getr(corrected_pos, window=rolling_window)

                # Early-rate flag (kept from your original logic)
                r_hour = float(time_index_to_numeric_hours([r_time])[0])
                r_flag = 2 if r_hour <= 1 else 0

                results.append({
                    "Sample": col,
                    "Baseline(time0)": baseline,
                    "K": k_val,
                    "K Index": k_time,
                    "K_tail_flag": K_tail_flag,
                    "r (1/h)": r_val,
                    "r Index": r_time,
                    "r_flag": r_flag,
                    "t Index": t_idx,
                    "t (h)": t_h
                })

            else:
                # Only one positive point
                if allow_single_point_k:
                    k_time, k_val = getK(corrected_nonan, window=rolling_window)
                    results.append({
                        "Sample": col,
                        "Baseline(time0)": baseline,
                        "K": k_val,
                        "K Index": k_time,
                        "K_tail_flag": np.nan,
                        "r (1/h)": np.nan,
                        "r Index": np.nan,
                        "r_flag": np.nan,
                        "t Index": t_idx,
                        "t (h)": t_h
                    })
                else:
                    excluded.append({
                        "Sample": col,
                        "Reason": "Not enough positive points after baseline correction",
                        "TotalPoints": total_points,
                        "PositivePoints": pos_points
                    })

        except Exception as e:
            excluded.append({
                "Sample": col,
                "Reason": f"Exception: {str(e)}",
                "TotalPoints": total_points,
                "PositivePoints": pos_points
            })
            continue

    return pd.DataFrame(results), pd.DataFrame(excluded)


def extract_sample_number(name):
    """Extract numeric ID from sample names for sorting."""
    m = re.search(r'(\d+)', str(name))
    return int(m.group(1)) if m else None


def main():
    # Read first sheet by default (no SHEET_NAME)
    df = pd.read_excel(INPUT_XLSX)
    df = ensure_time_index(df)

    expected_samples = list(df.columns)

    metrics_df, excluded_df = calculate_growth_parameters(
        df,
        rolling_window=ROLLING_WINDOW,
        allow_single_point_k=ALLOW_SINGLE_POINT_K,
        t_points=T_CONSECUTIVE_POINTS
    )

    # Sorting
    if not metrics_df.empty:
        nums = metrics_df["Sample"].map(extract_sample_number)
        max_num = nums.max() if pd.notnull(nums).any() else 0
        nums = nums.fillna(max_num + 1).astype(int)

        metrics_df = (
            metrics_df.assign(_SampleNum=nums)
            .sort_values("_SampleNum")
            .drop(columns=["_SampleNum"])
        )

    # Write Excel
    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        metrics_df.to_excel(writer, index=False, sheet_name='metrics')
        if not excluded_df.empty:
            excluded_df.to_excel(writer, index=False, sheet_name='excluded')

    # Console summary
    processed = set(metrics_df["Sample"]) if not metrics_df.empty else set()
    missing = sorted(list(set(expected_samples) - processed))

    print(f"Total samples(columns): {len(expected_samples)}")
    print(f"Successfully processed: {len(processed)}")
    print(f"Excluded: {len(missing)}")
    if missing:
        print("Excluded samples:", ", ".join(missing))
    print("Results saved to:", OUTPUT_XLSX)


if __name__ == "__main__":
    main()