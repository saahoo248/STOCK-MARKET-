def add_moving_averages(df):
    df = df.copy()

    df["ma20"] = df["close"].rolling(
        window=20
    ).mean()

    df["ma50"] = df["close"].rolling(
        window=50
    ).mean()

    return df
