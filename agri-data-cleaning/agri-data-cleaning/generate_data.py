"""Generate a synthetic, deliberately messy agribusiness dataset (AgriFarm Survey 2025-26).

All data is hypothetical and created only to demonstrate the cleaning strategy.
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
N = 600

districts = ["Nagpur", "Wardha", "Amravati", "Yavatmal"]
crops_raw = {
    "soybean": ["Soybean", "SOYA BEAN ", "soyabean", "Soya bean"],
    "cotton": ["Cotton", "cotton ", "COTTON", "Kapas"],
    "wheat": ["Wheat", "wheat ", "WHEAT", "gehu"],
}
base_yield = {"soybean": 12, "cotton": 10, "wheat": 28}  # q/ha (approx.)

rows = []
for i in range(N):
    crop = rng.choice(list(crops_raw))
    district = rng.choice(districts)
    area_ha = float(np.round(rng.gamma(2.0, 0.8) + 0.2, 2))
    unit = rng.choice(["ha", "acre", "bigha"], p=[0.5, 0.35, 0.15])
    factor = {"ha": 1.0, "acre": 0.4047, "bigha": 0.1333}[unit]
    area_val = round(area_ha / factor, 2)
    yld = max(1.0, rng.normal(base_yield[crop], 3))
    month = {"soybean": 7, "cotton": 6, "wheat": 11}[crop]
    day = int(rng.integers(1, 28))
    d = pd.Timestamp(2025, month, day)
    fmt = rng.choice(["%d/%m/%y", "%Y-%m-%d", "%d %b %Y"])
    rows.append({
        "farmer_id": f"F{100 + i}",
        "plot_id": f"P{rng.integers(1, 3)}",
        "season": {"soybean": "Kharif 2025", "cotton": "Kharif 2025", "wheat": "Rabi 2025"}[crop],
        "district": district.lower() if rng.random() < 0.2 else district,
        "crop": rng.choice(crops_raw[crop]),
        "area_value": area_val,
        "area_unit": unit,
        "sowing_date": d.strftime(fmt),
        "yield_q_ha": round(yld, 1),
        "soil_ph": round(float(rng.normal(7.6, 0.5)), 1),
        "rainfall_mm": round(float(rng.normal(620, 80)), 0),
        "fertiliser_kg_ha": round(float(rng.normal(80, 20)), 0),
    })

df = pd.DataFrame(rows)

# --- inject problems ---
miss = rng.choice(df.index, int(0.18 * N), replace=False)
df.loc[miss, "soil_ph"] = np.nan                       # missing pH
df.loc[rng.choice(df.index, 20, replace=False), "rainfall_mm"] = -999   # sensor fault code
df.loc[rng.choice(df.index, 15, replace=False), "yield_q_ha"] = np.nan   # missing target
df.loc[rng.choice(df.index, 6, replace=False), "yield_q_ha"] *= 10       # decimal-point errors
df.loc[rng.choice(df.index, 5, replace=False), "soil_ph"] = 15.0         # impossible pH
df.loc[rng.choice(df.index, 4, replace=False), "area_value"] = -1.0      # impossible area
df = pd.concat([df, df.sample(12, random_state=SEED)], ignore_index=True)  # exact duplicates
df["fertiliser_kg_ha"] = df["fertiliser_kg_ha"].astype(object)
df.loc[rng.choice(df.index, 10, replace=False), "fertiliser_kg_ha"] = "NA"

df.to_csv("data/raw/agrifarm_raw.csv", index=False)
print(f"Wrote data/raw/agrifarm_raw.csv with {len(df)} rows")
