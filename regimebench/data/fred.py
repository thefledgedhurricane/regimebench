"""FRED Macro Data Connector: USREC, VIXCLS, T10Y2Y, NFCI."""

import os
import pandas as pd
import numpy as np
from typing import Optional
from regimebench.data.provenance import record_provenance, load_cached_raw

def fetch_fred_series(
    series_id: str = "USREC", 
    api_key: Optional[str] = None, 
    allow_synthetic: bool = False
) -> pd.DataFrame:
    """Downloads series from FRED.
    
    Raises RuntimeError if fetch fails unless allow_synthetic=True is explicitly set.
    """
    dataset_name = f"fred_{series_id.lower()}"
    cached = load_cached_raw(dataset_name)
    if cached is not None and not cached.empty:
        return cached

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")

    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            data = fred.get_series(series_id)
            df = pd.DataFrame({series_id: data})
            record_provenance(dataset_name, f"FRED API ({series_id})", df)
            return df
        except Exception:
            pass

    try:
        from pandas_datareader import data as web
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        df = web.DataReader(series_id, 'fred', '1926-01-01', today_str)
        record_provenance(dataset_name, f"FRED Datareader ({series_id})", df)
        return df
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(f"FRED fetch failed for series '{series_id}' ({e}). Synthetic fallback disabled by default (allow_synthetic=False).") from e
        
        # Explicit synthetic fallback covering historical NBER recession windows (1926-today)
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        dates = pd.date_range(start='1926-01-01', end=today_str, freq='MS')
        rec = np.zeros(len(dates), dtype=int)
        rec[(dates.year >= 1929) & (dates.year <= 1933)] = 1  # Great Depression
        rec[(dates.year >= 1937) & (dates.year <= 1938)] = 1  # 1937 recession
        rec[(dates.year >= 1945) & (dates.year <= 1945)] = 1  # 1945 recession
        rec[(dates.year >= 1953) & (dates.year <= 1954)] = 1  # 1953 recession
        rec[(dates.year >= 1973) & (dates.year <= 1975)] = 1  # 1973 stagflation
        rec[(dates.year >= 1981) & (dates.year <= 1982)] = 1  # Volcker shock
        rec[(dates.year >= 1990) & (dates.year <= 1991)] = 1  # 1990 recession
        rec[(dates.year >= 2001) & (dates.year <= 2001)] = 1  # Dot-com crash
        rec[(dates.year >= 2008) & (dates.year <= 2009)] = 1  # GFC
        rec[(dates.year == 2020)] = 1                         # COVID-19 shock
        return pd.DataFrame({'USREC': rec}, index=dates)
