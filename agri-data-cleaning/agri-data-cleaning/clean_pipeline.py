"""Reproducible data cleaning and preprocessing pipeline for agribusiness data.

Run:  python clean_pipeline.py
Reads:   data/raw/agrifarm_raw.csv           (never modified)
Writes:  data/processed/agrifarm_clean.csv   (cleaned data)
         data/processed/agrifarm_model_ready.csv (encoded + scaled features)
         reports/cleaning_log.csv            (what each rule changed)
         reports/quality_before_after.csv    (data quality comparison)
"""
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# ---------------- configuration (kept out of the logic) ----------------
PLACEHOLDERS = [-999, -999.0, 9999, "NA", "na", "n/a", "?", ""]
TO_HA = {"ha": 1.0, "acre": 0.4047, "bigha": 0.1333}   # bigha factor is hypothetical
CROP_MAP = {
    "soybean": "soybean", "soya bean": "soybean", "soyabean": "soybean",
    "cotton": "cotton", "kapas": "cotton",
    "wheat": "wheat", "gehu": "wheat",
}
RANGES = {"soil_ph": (3, 10), "rainfall_mm": (0, 1500), "area_ha": (0.05, 50),
          "fertiliser_kg_ha": (0, 400)}
DATE_FORMATS = ["%d/%m/%y", "%Y-%m-%d", "%d %b %Y"]
CROP_YIELD_LIMIT = {"soybean": 40, "cotton": 30, "wheat": 70}   # agronomic upper limits (q/ha)

log = []


def record(rule, n, note=""):
    log.append({"rule": rule, "records_affected": int(n), "note": note})


def parse_date(s):
    for f in DATE_FORMATS:
        try:
            return pd.to_datetime(s, format=f)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def quality(df, label):
    return {
        "stage": label,
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells_pct": round(df.isna().mean().mean() * 100, 2),
    }


def main():
    raw = pd.read_csv("data/raw/agrifarm_raw.csv", dtype=str)
    q_before = quality(raw.replace(PLACEHOLDERS, np.nan), "before (raw, placeholders as missing)")
    df = raw.copy()

    # Step 3: placeholders and types
    n_ph = df.isin([str(p) for p in PLACEHOLDERS]).sum().sum()
    df = df.replace([str(p) for p in PLACEHOLDERS], np.nan)
    record("R01 placeholders -> missing", n_ph, "-999, NA, ?, blank")
    for c in ["area_value", "yield_q_ha", "soil_ph", "rainfall_mm", "fertiliser_kg_ha"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Step 4: standardise
    df["crop"] = df["crop"].str.strip().str.lower().map(CROP_MAP)
    df["district"] = df["district"].str.strip().str.title()
    record("R02 crop/district names standardised", len(df))
    df["sowing_date"] = df["sowing_date"].apply(parse_date)
    record("R03 dates parsed to ISO", df["sowing_date"].notna().sum(), f"{df['sowing_date'].isna().sum()} unparsed")
    df["area_ha"] = (df["area_value"] * df["area_unit"].map(TO_HA)).round(3)
    record("R04 area converted to hectare", df["area_ha"].notna().sum())
    df = df.drop(columns=["area_value", "area_unit"])

    # Step 5: duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    record("R05 exact duplicates removed", n0 - len(df))

    # Step 7a: domain-rule range checks -> set impossible values to missing
    for col, (lo, hi) in RANGES.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        df.loc[bad, col] = np.nan
        record(f"R06 range check {col} [{lo},{hi}]", bad.sum(), "set to missing")

    # Step 7b: crop-specific yield limit + group IQR (flag, do not delete)
    over = df["yield_q_ha"] > df["crop"].map(CROP_YIELD_LIMIT)
    df.loc[over, "yield_q_ha"] = np.nan
    record("R07 yield above agronomic limit", over.sum(), "set to missing (decimal-point errors)")
    grp = df.groupby("crop")["yield_q_ha"]
    q1, q3 = grp.transform(lambda s: s.quantile(.25)), grp.transform(lambda s: s.quantile(.75))
    iqr = q3 - q1
    df["yield_outlier_flag"] = ((df["yield_q_ha"] < q1 - 1.5 * iqr) | (df["yield_q_ha"] > q3 + 1.5 * iqr)).astype(int)
    record("R08 yield IQR outliers flagged (kept)", df["yield_outlier_flag"].sum())

    # Step 6: missing values (flag first, then group-wise median)
    for col in ["soil_ph", "rainfall_mm", "fertiliser_kg_ha", "area_ha"]:
        df[col + "_was_missing"] = df[col].isna().astype(int)
        n_miss = df[col].isna().sum()
        df[col] = df[col].fillna(df.groupby("district")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
        record(f"R09 impute {col} (district median)", n_miss)
    n_t = df["yield_q_ha"].isna().sum()
    df = df.dropna(subset=["yield_q_ha"])           # never impute the target
    record("R10 rows without target dropped", n_t)

    # Step 8: logical checks
    df["season_year_ok"] = (df["sowing_date"].dt.year == 2025).astype(int)
    record("R11 logical check: sowing year", (df["season_year_ok"] == 0).sum())

    q_after = quality(df.drop(columns=[c for c in df if c.endswith("_was_missing") or c.endswith("_flag") or c == "season_year_ok"]), "after (clean)")

    df.to_csv("data/processed/agrifarm_clean.csv", index=False)

    # Step 9-10: features, encoding, scaling (fit on train only)
    feat = df.copy()
    feat["sowing_month"] = feat["sowing_date"].dt.month
    feat["fert_per_rain"] = feat["fertiliser_kg_ha"] / feat["rainfall_mm"]
    feat = pd.get_dummies(feat, columns=["crop", "district", "season"], dtype=int)
    train = feat.sample(frac=0.8, random_state=SEED)
    test = feat.drop(train.index)
    num = ["area_ha", "soil_ph", "rainfall_mm", "fertiliser_kg_ha", "fert_per_rain"]
    mu, sd = train[num].mean(), train[num].std()
    for part in (train, test):
        part[num] = (part[num] - mu) / sd          # parameters learned on train only
    train = train.assign(split="train")
    test = test.assign(split="test")
    out = pd.concat([train, test]).drop(columns=["sowing_date", "farmer_id", "plot_id"])
    out.to_csv("data/processed/agrifarm_model_ready.csv", index=False)

    pd.DataFrame(log).to_csv("reports/cleaning_log.csv", index=False)
    pd.DataFrame([q_before, q_after]).to_csv("reports/quality_before_after.csv", index=False)
    print(pd.DataFrame(log).to_string(index=False))
    print()
    print(pd.DataFrame([q_before, q_after]).to_string(index=False))


if __name__ == "__main__":
    main()
