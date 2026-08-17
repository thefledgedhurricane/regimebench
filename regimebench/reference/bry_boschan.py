"""Corrected Pagan-Sossounov / Bry-Boschan Turning Point Dating Algorithm.

Implements monthly Pagan-Sossounov (2003) / Bry-Boschan (1971) turning point rules:
- 5-month symmetric window local peak and trough detection
- Enforced 20% peak-to-trough drawdown threshold for Bear market phase validation
- Enforced 6-month minimum phase duration
"""

import numpy as np
import pandas as pd

def bry_boschan_monthly_dating(
    monthly_prices: pd.Series,
    min_phase_months: int = 6,
    min_cycle_months: int = 15,
    amplitude_pct: float = 0.20
) -> pd.Series:
    """Computes Pagan-Sossounov / Bry-Boschan turning point dating on monthly asset prices."""
    p = monthly_prices.values
    dates = monthly_prices.index
    n = len(p)
    regimes = np.ones(n, dtype=int)  # 1 = Bull, 0 = Bear
    
    # 1. 5-month symmetric window local peak and trough detection
    peaks = []
    troughs = []
    for i in range(5, n - 5):
        window = p[i-5 : i+6]
        if p[i] == np.max(window):
            peaks.append(i)
        if p[i] == np.min(window):
            troughs.append(i)
            
    # Combine and sort turns
    turns = []
    for pk in peaks:
        turns.append((pk, 'peak', p[pk]))
    for tr in troughs:
        turns.append((tr, 'trough', p[tr]))
    turns.sort(key=lambda x: x[0])
    
    def enforce_alternation(turns):
        if not turns: return []
        alt = [turns[0]]
        for i in range(1, len(turns)):
            if turns[i][1] == alt[-1][1]:
                # same type, keep the more extreme one
                if alt[-1][1] == 'peak':
                    if turns[i][2] > alt[-1][2]:
                        alt[-1] = turns[i]
                else: # trough
                    if turns[i][2] < alt[-1][2]:
                        alt[-1] = turns[i]
            else:
                alt.append(turns[i])
        return alt

    turns = enforce_alternation(turns)
    
    while True:
        to_remove = set()
        
        # Phase duration (6 months)
        for i in range(len(turns) - 1):
            if turns[i+1][0] - turns[i][0] < min_phase_months:
                to_remove.add(i)
                to_remove.add(i+1)
                
        # Cycle duration (15 months)
        for i in range(len(turns) - 2):
            if turns[i+2][0] - turns[i][0] < min_cycle_months:
                if turns[i][1] == 'peak':
                    if turns[i][2] < turns[i+2][2]:
                        to_remove.add(i)
                        to_remove.add(i+1)
                    else:
                        to_remove.add(i+1)
                        to_remove.add(i+2)
                else:
                    if turns[i][2] > turns[i+2][2]:
                        to_remove.add(i)
                        to_remove.add(i+1)
                    else:
                        to_remove.add(i+1)
                        to_remove.add(i+2)
                        
        if to_remove:
            turns = [t for j, t in enumerate(turns) if j not in to_remove]
            turns = enforce_alternation(turns)
        else:
            break

    # Amplitude check: Bear phase must drop >= 20%
    while True:
        to_remove = set()
        for i in range(len(turns) - 1):
            if turns[i][1] == 'peak' and turns[i+1][1] == 'trough':
                peak_val = turns[i][2]
                trough_val = turns[i+1][2]
                if (peak_val - trough_val) / peak_val < amplitude_pct:
                    to_remove.add(i)
                    to_remove.add(i+1)
                    
        if to_remove:
            turns = [t for j, t in enumerate(turns) if j not in to_remove]
            turns = enforce_alternation(turns)
        else:
            break

    # Endpoint turn elimination: remove turns within min_phase_months of start/end
    turns = [t for t in turns if t[0] >= min_phase_months and (n - 1 - t[0]) >= min_phase_months]
    turns = enforce_alternation(turns)

    # Convert to regimes
    for i in range(len(turns) - 1):
        if turns[i][1] == 'peak' and turns[i+1][1] == 'trough':
            regimes[turns[i][0]:turns[i+1][0]+1] = 0

    return pd.Series(regimes, index=dates, name="Bry_Boschan_Bear")

def bry_boschan_daily_dating(daily_prices: pd.Series) -> pd.Series:
    """Daily wrapper aggregating price level to monthly, computing Pagan-Sossounov, and re-expanding."""
    monthly_p = daily_prices.resample('ME').last().dropna()
    monthly_regimes = bry_boschan_monthly_dating(monthly_p)
    
    df_daily = daily_prices.to_frame('price')
    df_daily['month'] = df_daily.index.to_period('M')
    
    # Map monthly regimes back to daily rows
    monthly_regimes_df = monthly_regimes.to_frame('Bry_Boschan_Bear')
    monthly_regimes_df['month'] = monthly_regimes_df.index.to_period('M')
    
    merged = df_daily.merge(monthly_regimes_df[['month', 'Bry_Boschan_Bear']].drop_duplicates('month'), on='month', how='left')
    return pd.Series(merged['Bry_Boschan_Bear'].ffill().bfill().fillna(1).astype(int).values, index=daily_prices.index, name="Bry_Boschan_Bear")
