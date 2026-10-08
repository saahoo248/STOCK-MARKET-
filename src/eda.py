def run_eda(df):
    print("\n========================================")
    print("EXPLORATORY DATA ANALYSIS")
    print("========================================")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistics:")
    print(
        df[
            ["open", "high", "low", "close", "volume"]
        ].describe()
    )

    print("\nDate Range:")
    print(df["date"].min(), "to", df["date"].max())

    if "Name" in df.columns:
        print("\nCompanies:")
        print(df["Name"].nunique())
    elif "name" in df.columns:
        print("\nCompanies:")
        print(df["name"].nunique())
