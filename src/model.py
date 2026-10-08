from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(df):
    data = df[
        ["date", "close"]
    ].dropna().copy()

    # Convert date to numeric day feature
    data["day"] = (
        data["date"] - data["date"].min()
    ).dt.days

    X = data[["day"]]
    y = data["close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return (
        model,
        X_test,
        y_test,
        predictions
    )
