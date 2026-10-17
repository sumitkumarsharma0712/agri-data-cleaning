"""Validation tests for the cleaned output.

Run after the pipeline:  python tests/test_pipeline.py   (or: pytest -q)
"""
import pandas as pd

CLEAN = "data/processed/agrifarm_clean.csv"
MODEL = "data/processed/agrifarm_model_ready.csv"


def load():
    return pd.read_csv(CLEAN, parse_dates=["sowing_date"])


def test_no_duplicates():
    df = load()
    assert not df.duplicated(["farmer_id", "plot_id", "season"]).any()


def test_no_missing_in_key_columns():
    df = load()
    for c in ["crop", "district", "sowing_date", "yield_q_ha", "soil_ph", "rainfall_mm", "fertiliser_kg_ha", "area_ha"]:
        assert df[c].notna().all(), f"missing values in {c}"


def test_crop_names_standardised():
    assert set(load()["crop"]) <= {"soybean", "cotton", "wheat"}


def test_value_ranges():
    df = load()
    assert df["soil_ph"].between(3, 10).all()
    assert df["area_ha"].between(0.05, 50).all()
    assert df["rainfall_mm"].between(0, 1500).all()
    assert (df["yield_q_ha"] > 0).all()


def test_yield_within_agronomic_limits():
    df = load()
    limit = df["crop"].map({"soybean": 40, "cotton": 30, "wheat": 70})
    assert (df["yield_q_ha"] <= limit).all()


def test_dates_are_iso_and_valid():
    assert load()["sowing_date"].dt.year.eq(2025).all()


def test_missing_flags_exist():
    df = load()
    assert "soil_ph_was_missing" in df.columns and df["soil_ph_was_missing"].sum() > 0


def test_model_ready_split_and_scaling():
    m = pd.read_csv(MODEL)
    assert set(m["split"]) == {"train", "test"}
    tr = m[m.split == "train"]
    assert abs(tr["rainfall_mm"].mean()) < 1e-6          # scaled with train statistics
    assert 0.75 < len(tr) / len(m) < 0.85


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
