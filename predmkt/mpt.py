"""Atlanta Fed Market Probability Tracker loader.

The MPT publishes daily implied probabilities for FOMC outcomes derived from
3-month SOFR options. Distributed as one xlsx with a `DATA` sheet:

    columns: date, reference_start, target_range, field, value

`reference_start` is the FOMC meeting date being forecast.
`field` includes 'Prob: cut', 'Prob: hike', and bucketed
'Prob: <lo>bps - <hi>bps' (the post-meeting target range).
`target_range` is metadata for context, not used for lookup.

URL:
    https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cenfis/market-probability-tracker/mpt_histdata.xlsx
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

MPT_URL = (
    "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/"
    "cenfis/market-probability-tracker/mpt_histdata.xlsx"
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PATH = DATA_DIR / "mpt_histdata.xlsx"

_BUCKET_RE = re.compile(r"^Prob:\s*(\d+)bps\s*-\s*(\d+)bps$")


def download(path: Path = DEFAULT_PATH, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    r = requests.get(MPT_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


@dataclass
class MPT:
    """Long-format MPT data with helpers."""

    df: pd.DataFrame  # columns: date, reference_start, field, value (target_range dropped — redundant)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "MPT":
        if not path.exists():
            download(path)
        raw = pd.read_excel(path, sheet_name="DATA")
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize("UTC").dt.normalize()
        raw["reference_start"] = pd.to_datetime(raw["reference_start"]).dt.tz_localize("UTC").dt.normalize()
        keep = ["date", "reference_start", "field", "value"]
        return cls(df=raw[keep].copy())

    @property
    def meetings(self) -> list[pd.Timestamp]:
        return sorted(self.df["reference_start"].unique())

    def for_meeting(self, reference_start: pd.Timestamp) -> pd.DataFrame:
        ref = pd.Timestamp(reference_start).tz_localize("UTC") if pd.Timestamp(reference_start).tzinfo is None else pd.Timestamp(reference_start)
        return self.df[self.df["reference_start"] == ref.normalize()].copy()

    def bucket_prob_series(
        self, reference_start: pd.Timestamp, lo_bps: int, hi_bps: int
    ) -> pd.Series:
        """Daily time series of P(target ∈ [lo_bps, hi_bps]) for one meeting."""
        field = f"Prob: {lo_bps}bps - {hi_bps}bps"
        sub = self.for_meeting(reference_start)
        sub = sub[sub["field"] == field].sort_values("date")
        return pd.Series(sub["value"].to_numpy(), index=pd.DatetimeIndex(sub["date"]), name=field)

    def cut_prob_series(self, reference_start: pd.Timestamp) -> pd.Series:
        sub = self.for_meeting(reference_start)
        sub = sub[sub["field"] == "Prob: cut"].sort_values("date")
        return pd.Series(sub["value"].to_numpy(), index=pd.DatetimeIndex(sub["date"]), name="Prob: cut")

    def hike_prob_series(self, reference_start: pd.Timestamp) -> pd.Series:
        sub = self.for_meeting(reference_start)
        sub = sub[sub["field"] == "Prob: hike"].sort_values("date")
        return pd.Series(sub["value"].to_numpy(), index=pd.DatetimeIndex(sub["date"]), name="Prob: hike")

    def available_buckets(self, reference_start: pd.Timestamp) -> list[tuple[int, int]]:
        sub = self.for_meeting(reference_start)
        out = []
        for f in sub["field"].unique():
            m = _BUCKET_RE.match(str(f))
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
        return sorted(set(out))
