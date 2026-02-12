import pandas as pd


def test_pref_columns(tmp_path):
    df = pd.DataFrame(
        [{"image": "a.jpg", "prompt": "p", "chosen": "{}", "rejected": "{}"}]
    )
    out = tmp_path / "x.parquet"
    df.to_parquet(out, index=False)
    loaded = pd.read_parquet(out)
    assert {"image", "prompt", "chosen", "rejected"}.issubset(set(loaded.columns))
