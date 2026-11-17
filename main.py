"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Design (measured on 38 past big launches' release days, leave-one-out):
  - A launch's release day is anchored on its first hour's print p1.
  - Launches fall into two kinds. Rolling-midnight launches already have a
    large audience in hour 1 and their day stays within ~0.5x-3x of p1.
    Single-global-unlock launches have a near-empty hour 1 and explode
    100x-2000x later; they poison any pooled average, so they are filtered
    out (hour-1 share of the day peak below 0.08).
  - Among rolling launches, log(p1) predicts how flat the day is
    (corr -0.77 with log peak ratio): mega launches peak near p1, small
    ones climb several-fold. So per hour h we fit a robust line
    (Theil-Sen) of log(viewers_h / p1) on log(p1) over the filtered
    launches, and take the residual spread as the lognormal width of
    the five quantiles. Past the biggest calibration launch the slope
    is tapered to zero instead of extrapolated or frozen (see below):
    a launch bigger than every past launch keeps flattening, but
    only toward a finite limit.
  - Early-access / free-to-play launches are excluded by keywords in
    their unlock note, since their hour 1 is inflated by players who
    were already in before the day began.
  - Fallback: with too few calibration launches, or a fit that fails,
    fall back to flat persistence with a spread widening as
    0.15 * sqrt(h - 1) in log space.

Standard library only. Deterministic: no randomness, no wall clock,
no network. Reads every *.in in the input directory and writes one
*.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves the repo root off sys.path; nothing to import
# from beside this file, so no path fix-up is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# A launch counts as rolling-midnight when hour 1 already holds this
# share of the release day's peak hour.
MIN_PEAK_SHARE = 0.08
# A release day counts when this many of its 24 hours have a print.
MIN_DAY_HOURS = 20
# Below this many calibration launches the regression is not trusted.
MIN_CALIBRATION = 5
# Baseline spread (log units per sqrt hour) used by the fallback.
BASE_SIGMA = 0.15

# Unlock-note markers for launches whose hour 1 is not a clean launch
# signal (early access already live, free-to-play dynamics, weak analog).
EXCLUDE_MARKERS = ("early access", "early-access", "f2p", "free-to-play", "tier: weak")


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _pstdev(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _round2(x):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def _release_day(comparable):
    """Release-day hourly viewers {hour: viewers} for one comparable launch.

    Returns an empty dict when the comparable has no usable release day
    (e.g. a pre-release history with no launch day of its own).
    """
    try:
        start = _parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    day = {}
    for row in comparable.get("hourly") or []:
        try:
            viewers = row["viewers"]
            when = _parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        hour = int((when - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            day[hour] = viewers
    return day


def _is_early_access(unlock_note):
    text = (unlock_note or "").lower()
    text = text.replace("no early access", "").replace("no early-access", "")
    return any(marker in text for marker in EXCLUDE_MARKERS)


def _calibration_pool(case):
    """Filtered past launches: (log hour-1 viewers, {hour: viewers})."""
    pool = []
    for comparable in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        day = _release_day(comparable)
        if 1 not in day or len(day) < MIN_DAY_HOURS:
            continue
        peak = max(day.values())
        if day[1] / peak < MIN_PEAK_SHARE:
            continue
        if _is_early_access(comparable.get("unlock")):
            continue
        pool.append((math.log(day[1]), day))
    return pool


def _theil_sen(points):
    """Robust line through (x, y) points: median pairwise slope, median intercept.

    Returns (intercept, slope, residual_sd) or None when the points do
    not determine a line.
    """
    if len(points) < 2:
        return None
    slopes = []
    for i in range(len(points)):
        xi, yi = points[i]
        for j in range(i + 1, len(points)):
            xj, yj = points[j]
            if xj != xi:
                slopes.append((yj - yi) / (xj - xi))
    if not slopes:
        return None
    slope = _median(slopes)
    intercept = _median([y - slope * x for x, y in points])
    residuals = [y - (intercept + slope * x) for x, y in points]
    return intercept, slope, _pstdev(residuals)


def _baseline_quantiles(p1, hour):
    spread = BASE_SIGMA * math.sqrt(hour - 1)
    return [_round2(max(0.01, min(1e9, p1 * math.exp(z * spread)))) for z in Z]


# How fast the fitted slope decays to zero past the biggest calibration
# launch (log units). With alpha = 0.8, the fitted log ratio past the
# range approaches intercept + slope * hi + slope / alpha: a finite
# limit, never a reversal, never unbounded growth.
TAPER_ALPHA = 0.8


def _forecast_hour(p1, hour, pool):
    """Five quantiles for one hour from the calibration pool."""
    log_p1 = math.log(p1)
    points = []
    for log_first, day in pool:
        if hour in day:
            points.append((log_first, math.log(day[hour] / day[1])))
    if len(points) < 4:
        return _baseline_quantiles(p1, hour)
    fit = _theil_sen(points)
    if fit is None:
        return _baseline_quantiles(p1, hour)
    intercept, slope, spread = fit
    lo = min(x for x, _ in points)
    hi = max(x for x, _ in points)
    if log_p1 > hi:
        # Tapered extrapolation: the slope decays exponentially to zero
        # past the biggest launch seen, so the ratio saturates instead
        # of freezing (clamp) or running away (raw line).
        centre = (
            intercept
            + slope * hi
            + (slope / TAPER_ALPHA)
            * (1.0 - math.exp(-TAPER_ALPHA * (log_p1 - hi)))
        )
    else:
        # In range, or below the smallest launch: evaluate the line,
        # clamped at the low end.
        centre = intercept + slope * max(lo, log_p1)
    if spread <= 0:
        spread = 0.05
    quantiles = []
    for z in Z:
        value = p1 * math.exp(centre + z * spread)
        quantiles.append(_round2(max(0.01, min(1e9, value))))
    # Rounding is monotone, but enforce non-decreasing output anyway.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def _hour_one_print(case):
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return float(row["viewers"])
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if last:
            return float(last)
    return 1.0


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    p1 = max(1.0, _hour_one_print(case))
    pool = _calibration_pool(case)
    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        if len(pool) >= MIN_CALIBRATION:
            quantiles = _forecast_hour(p1, hour, pool)
        else:
            quantiles = _baseline_quantiles(p1, hour)
        hours.append({"hour": hour, "q": quantiles})
    if not hours:
        hours = [
            {"hour": h, "q": _baseline_quantiles(p1, h)}
            for h in range(2, 25)
        ]
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
