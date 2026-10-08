import pandas as pd

def clean_data(df):
    df = df.copy()

    # Normalize names again for safety
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required = ["date", "open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Convert numeric columns
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows where essential values are missing
    df = df.dropna(
        subset=["date", "close"]
    )

    # Fill OHLC missing values using forward/backward fill
    for col in ["open", "high", "low"]:
        df[col] = df[col].ffill().bfill()

    # Fill remaining volume values if any
    df["volume"] = df["volume"].fillna(0)

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    return df
