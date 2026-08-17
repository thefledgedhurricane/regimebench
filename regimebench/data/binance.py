"""Binance Public Data Connector: 1-Minute BTCUSDT and ETHUSDT Klines (data.binance.vision)."""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from typing import Optional
from regimebench.data.provenance import record_provenance, load_cached_raw

BINANCE_PUBLIC_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

def fetch_binance_klines(
    symbol: str = "BTCUSDT", 
    year: int = 2024, 
    month: int = 1,
    allow_synthetic: bool = False
) -> pd.DataFrame:
    """Downloads 1-minute klines from Binance Public Data repository (data.binance.vision).
    
    Raises RuntimeError if download fails unless allow_synthetic=True is explicitly set.
    """
    month_str = f"{month:02d}"
    dataset_name = f"binance_{symbol.lower()}_{year}_{month_str}"
    
    cached = load_cached_raw(dataset_name)
    if cached is not None and not cached.empty:
        return cached

    url = f"{BINANCE_PUBLIC_BASE_URL}/{symbol}/1m/{symbol}-1m-{year}-{month_str}.zip"
    print(f"Downloading Binance 1m data for {symbol} ({year}-{month_str}) from {url}...")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_bytes = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, header=None)
                
        cols = [
            "Open_Time", "Open", "High", "Low", "Close", "Volume",
            "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
            "Taker_Buy_Base_Asset_Volume", "Taker_Buy_Quote_Asset_Volume", "Ignore"
        ]
        df.columns = cols[:df.shape[1]]
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        df['Returns'] = df['Close'].pct_change()
        df['Realized_Vol_60m'] = df['Returns'].rolling(60, min_periods=1).std() * np.sqrt(525600)
        df.set_index('Open_Time', inplace=True)
        res = df.dropna()
        record_provenance(dataset_name, url, res, payload_bytes=zip_bytes)
        print(f"   Successfully fetched {len(res):,} 1-minute rows for {symbol} ({year}-{month_str}).")
        return res
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(f"Binance 1-minute download failed for {symbol} ({year}-{month_str}) from {url}: {e}. Synthetic fallback disabled by default (allow_synthetic=False).") from e

        print(f"⚠️ Binance download error ({e}); returning explicit synthetic fallback.")
        dates = pd.date_range(start=f'{year}-{month_str}-01', periods=43200, freq='1min')
        np.random.seed(42)
        r = np.random.normal(0.0, 0.0008, size=len(dates))
        price = 45000.0 * np.exp(np.cumsum(r))
        df = pd.DataFrame({'Close': price, 'Returns': r}, index=dates)
        df['Realized_Vol_60m'] = df['Returns'].rolling(60, min_periods=1).std() * np.sqrt(525600)
        return df.dropna()
