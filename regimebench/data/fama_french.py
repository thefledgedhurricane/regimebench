"""Fama-French Data Library Connector: Daily Market Excess Returns (1926-2026, 26,274 trading days)."""

import os
import pandas as pd
import numpy as np
from typing import Optional
from regimebench.data.provenance import record_provenance, load_cached_raw

def fetch_fama_french_daily(allow_synthetic: bool = False) -> pd.DataFrame:
    """Downloads daily Fama-French Market Excess Returns across full century (1926-2026).
    
    Raises RuntimeError if fetch fails unless allow_synthetic=True is explicitly set.
    """
    # 1. Try local raw cache
    cached = load_cached_raw("fama_french_daily")
    if cached is not None and not cached.empty:
        return cached

    # 2. Try network download
    try:
        from pandas_datareader import data as web
        ff_dict = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1926-07-01')
        ff = ff_dict[0]
        ff.index = pd.to_datetime(ff.index.astype(str))
        ff = ff.rename(columns={'Mkt-RF': 'Mkt_RF', 'RF': 'RF'})
        ff['Returns'] = ff['Mkt_RF'] / 100.0
        res = ff[['Returns', 'Mkt_RF', 'SMB', 'HML', 'RF']]
        record_provenance("fama_french_daily", "F-F_Research_Data_Factors_daily (Dartmouth)", res)
        return res
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(f"Fama-French network fetch failed ({e}). Synthetic fallback disabled by default (allow_synthetic=False).") from e
        
        # Explicit synthetic fallback
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        dates = pd.date_range(start='1926-07-01', end=today_str, freq='B')
        np.random.seed(42)
        r = np.random.normal(0.0004, 0.01, size=len(dates))
        df = pd.DataFrame({'Returns': r, 'Mkt_RF': r * 100}, index=dates)
        return df
