"""
Synthetic Predictive Maintenance Dataset Generator
----------------------------------------------------
Simulates 4 sensors on an industrial machine:
  - temperature   (°C)
  - vibration     (mm/s)
  - pressure      (bar)
  - rpm           (rotations/min)

Three failure modes are injected:
  1. Overheat      – temperature drifts up then spikes
  2. Bearing wear  – vibration slowly increases, rpm drops
  3. Pressure drop – pressure falls sharply before failure

Label: 1 if failure occurs within the next 30 timesteps, else 0.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)

# ── constants ──────────────────────────────────────────────────────────────
TIMESTEPS      = 50_000   # total time steps per run
SAMPLE_HZ      = 1        # 1 sample per second
FAILURE_HORIZON = 30      # label=1 if failure within next 30 steps
SEQ_LEN        = 60       # lookback window fed to models
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# test = remaining 0.15

# ── baseline signal parameters ─────────────────────────────────────────────
BASELINES = dict(
    temperature = 75.0,   # °C
    vibration   =  1.2,   # mm/s
    pressure    =  5.0,   # bar
    rpm         = 1450.0, # RPM
)
NOISE = dict(
    temperature = 0.8,
    vibration   = 0.15,
    pressure    = 0.05,
    rpm         = 8.0,
)


def _base_signal(n):
    """Stationary baseline with realistic correlated drift."""
    t   = np.arange(n)
    temp = (BASELINES['temperature']
            + 2.0 * np.sin(2 * np.pi * t / 3600)   # hourly thermal cycle
            + np.random.normal(0, NOISE['temperature'], n))
    vib  = (BASELINES['vibration']
            + 0.1 * np.sin(2 * np.pi * t / 600)
            + np.random.normal(0, NOISE['vibration'], n))
    pres = (BASELINES['pressure']
            + 0.02 * np.sin(2 * np.pi * t / 1800)
            + np.random.normal(0, NOISE['pressure'], n))
    rpm  = (BASELINES['rpm']
            + 5.0 * np.sin(2 * np.pi * t / 900)
            + np.random.normal(0, NOISE['rpm'], n))
    return temp, vib, pres, rpm


def _inject_overheat(temp, vib, pres, rpm, start, duration=200):
    """Temperature ramp + spike. Mild vibration uptick."""
    end = min(start + duration, len(temp))
    ramp = np.linspace(0, 25, end - start)
    spike = np.random.normal(0, 2.5, end - start)
    temp[start:end] += ramp + spike
    vib[start:end]  += np.linspace(0, 0.4, end - start)
    return start + duration  # next available index


def _inject_bearing_wear(temp, vib, pres, rpm, start, duration=300):
    """Gradual vibration rise, rpm sag, slight heat."""
    end = min(start + duration, len(temp))
    vib[start:end]  += np.linspace(0, 3.5, end - start) + np.random.normal(0, 0.2, end - start)
    rpm[start:end]  -= np.linspace(0, 80, end - start)
    temp[start:end] += np.linspace(0, 8, end - start)
    return start + duration


def _inject_pressure_drop(temp, vib, pres, rpm, start, duration=120):
    """Sharp pressure fall, rpm instability."""
    end = min(start + duration, len(temp))
    pres[start:end] -= np.linspace(0, 3.5, end - start) + np.random.normal(0, 0.1, end - start)
    rpm[start:end]  += np.random.normal(0, 30, end - start)
    return start + duration


def generate(n=TIMESTEPS, failure_rate=0.08, seed=SEED):
    np.random.seed(seed)
    temp, vib, pres, rpm = _base_signal(n)
    failure_times = []

    failure_funcs = [_inject_overheat, _inject_bearing_wear, _inject_pressure_drop]
    i = 500
    while i < n - 500:
        if np.random.rand() < failure_rate:
            fn = np.random.choice(failure_funcs)
            failure_end = fn(temp, vib, pres, rpm, i)
            failure_times.append(failure_end)
            i = failure_end + np.random.randint(300, 800)  # cool-down
        else:
            i += np.random.randint(50, 150)

    # ── build label vector ─────────────────────────────────────────────────
    labels = np.zeros(n, dtype=int)
    for ft in failure_times:
        start_label = max(0, ft - FAILURE_HORIZON)
        labels[start_label:ft] = 1

    df = pd.DataFrame({
        'timestamp':   pd.date_range('2024-01-01', periods=n, freq='1s'),
        'temperature': np.clip(temp, 50, 130),
        'vibration':   np.clip(vib,   0, 10),
        'pressure':    np.clip(pres,   0, 8),
        'rpm':         np.clip(rpm,  900, 1800),
        'label':       labels,
    })
    return df


def make_sequences(df, seq_len=SEQ_LEN, features=None):
    """Slide a window over the dataframe → (X, y) numpy arrays."""
    if features is None:
        features = ['temperature', 'vibration', 'pressure', 'rpm']
    X, y = [], []
    vals = df[features].values
    lbls = df['label'].values
    for i in range(len(df) - seq_len):
        X.append(vals[i:i + seq_len])
        y.append(lbls[i + seq_len])   # label at the END of the window
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def split_and_save(df, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))

    df.iloc[:t1].to_csv(f'{out_dir}/train_raw.csv', index=False)
    df.iloc[t1:t2].to_csv(f'{out_dir}/val_raw.csv',   index=False)
    df.iloc[t2:].to_csv(f'{out_dir}/test_raw.csv',    index=False)

    for split, part in [('train', df.iloc[:t1]),
                        ('val',   df.iloc[t1:t2]),
                        ('test',  df.iloc[t2:])]:
        X, y = make_sequences(part.reset_index(drop=True))
        np.save(f'{out_dir}/{split}_X.npy', X)
        np.save(f'{out_dir}/{split}_y.npy', y)
        pos = y.sum(); neg = len(y) - pos
        print(f'{split:5s}: {len(X):6d} sequences | '
              f'failures={pos} ({100*pos/len(y):.1f}%) | normal={neg}')


if __name__ == '__main__':
    print('Generating synthetic sensor data …')
    df = generate()
    split_and_save(df)
    print('\nSaved to data/ — run train.py next.')
