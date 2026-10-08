import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Rename common alternatives
    rename_map = {
        "timestamp": "date",
        "datetime": "date",
        "adj close": "close",
        "adj_close": "close",
    }

    df = df.rename(columns=rename_map)

    return df
