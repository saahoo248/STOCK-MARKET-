from flask import Flask, render_template, request, redirect, url_for, send_file
from pathlib import Path
from license_util import verify_license, get_saved_license
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

BASE = Path(__file__).resolve().parent

DATA_FILE = BASE / "data" / "raw" / "stock_data.csv"
CHART_DIR = BASE / "outputs" / "charts" / "companies"


def get_license():
    key = get_saved_license()

    if not key:
        return False, None

    valid, data = verify_license(key)

    if not valid:
        return False, None

    return True, data


def load_dataset():
    if not DATA_FILE.exists():
        return None, "Dataset not found."

    df = pd.read_csv(DATA_FILE)

    # Normalize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Common alternative column names
    rename_map = {
        "timestamp": "date",
        "datetime": "date",
        "adj close": "close",
        "adj_close": "close"
    }

    df = df.rename(columns=rename_map)

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        return None, (
            "Required columns missing: "
            + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Company name
    if "name" not in df.columns:
        df["name"] = "All Companies"

    df["name"] = (
        df["name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=["date", "close"]
    )

    for col in [
        "open",
        "high",
        "low"
    ]:
        df[col] = df[col].ffill().bfill()

    df["volume"] = df["volume"].fillna(0)

    df = (
        df
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df, None


def analyze_company(df, company, start_date, end_date):

    if company != "ALL":
        work = df[
            df["name"] == company
        ].copy()
    else:
        work = df.copy()

    if start_date:
        start = pd.to_datetime(
            start_date,
            errors="coerce"
        )

        if not pd.isna(start):
            work = work[
                work["date"] >= start
            ]

    if end_date:
        end = pd.to_datetime(
            end_date,
            errors="coerce"
        )

        if not pd.isna(end):
            work = work[
                work["date"] <= end
            ]

    work = (
        work
        .sort_values("date")
        .reset_index(drop=True)
    )

    if work.empty:
        return {
            "error": "No data available for this selection."
        }

    # Moving averages
    work["ma20"] = (
        work["close"]
        .rolling(20)
        .mean()
    )

    work["ma50"] = (
        work["close"]
        .rolling(50)
        .mean()
    )

    # Basic statistics
    latest_close = float(
        work["close"].iloc[-1]
    )

    highest_close = float(
        work["close"].max()
    )

    lowest_close = float(
        work["close"].min()
    )

    avg_close = float(
        work["close"].mean()
    )

    avg_volume = float(
        work["volume"].mean()
    )

    latest_ma20 = work["ma20"].dropna()

    latest_ma50 = work["ma50"].dropna()

    ma20_value = (
        float(latest_ma20.iloc[-1])
        if not latest_ma20.empty
        else None
    )

    ma50_value = (
        float(latest_ma50.iloc[-1])
        if not latest_ma50.empty
        else None
    )

    # -------------------------------------------------
    # LINEAR REGRESSION BASELINE
    # -------------------------------------------------

    model_metrics = {
        "MAE": None,
        "RMSE": None
    }

    predictions = pd.DataFrame()

    model_data = work[
        ["date", "close"]
    ].dropna().copy()

    if len(model_data) >= 10:

        model_data["day"] = (
            model_data["date"]
            - model_data["date"].min()
        ).dt.days

        X = model_data[["day"]]
        y = model_data["close"]

        split = int(
            len(model_data) * 0.8
        )

        if split > 0 and split < len(model_data):

            X_train = X.iloc[:split]
            X_test = X.iloc[split:]

            y_train = y.iloc[:split]
            y_test = y.iloc[split:]

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                pred
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    pred
                )
            )

            model_metrics = {
                "MAE": float(mae),
                "RMSE": float(rmse)
            }

            predictions = pd.DataFrame({
                "Date": X_test.index.map(
                    lambda x:
                    model_data.loc[x, "date"]
                ),
                "Actual Close": y_test.values,
                "Predicted Close": pred
            })

    # -------------------------------------------------
    # CHART
    # -------------------------------------------------

    safe_name = (
        company
        if company != "ALL"
        else "all_companies"
    )

    safe_name = "".join(
        c if (
            c.isalnum()
            or c in (
                "-",
                "_"
            )
        )
        else "_"
        for c in safe_name
    )

    chart_path = (
        CHART_DIR
        / f"{safe_name}.png"
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        work["date"],
        work["close"],
        label="Close Price",
        linewidth=1.3
    )

    plt.plot(
        work["date"],
        work["ma20"],
        label="20-Day MA",
        linewidth=1.5
    )

    plt.plot(
        work["date"],
        work["ma50"],
        label="50-Day MA",
        linewidth=1.5
    )

    title_name = (
        company
        if company != "ALL"
        else "All Companies"
    )

    plt.title(
        f"Stock Market Analysis - {title_name}"
    )

    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend()
    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        chart_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "error": None,
        "company": title_name,
        "rows": len(work),
        "start": work["date"].min().strftime(
            "%Y-%m-%d"
        ),
        "end": work["date"].max().strftime(
            "%Y-%m-%d"
        ),
        "latest_close": latest_close,
        "highest_close": highest_close,
        "lowest_close": lowest_close,
        "avg_close": avg_close,
        "avg_volume": avg_volume,
        "ma20": ma20_value,
        "ma50": ma50_value,
        "mae": model_metrics["MAE"],
        "rmse": model_metrics["RMSE"],
        "chart": chart_path.name,
        "predictions": predictions.tail(
            10
        ).to_dict(
            orient="records"
        )
    }


@app.route("/")
def home():

    valid, license_data = get_license()

    if not valid:
        return redirect(
            url_for("activate")
        )

    df, error = load_dataset()

    if error:
        return render_template(
            "dashboard.html",
            license=license_data,
            error=error,
            companies=[],
            result=None
        )

    companies = sorted(
        df["name"]
        .dropna()
        .unique()
        .tolist()
    )

    company = request.args.get(
        "company",
        "ALL"
    )

    start_date = request.args.get(
        "start_date",
        ""
    )

    end_date = request.args.get(
        "end_date",
        ""
    )

    if (
        company != "ALL"
        and company not in companies
    ):
        company = "ALL"

    result = analyze_company(
        df,
        company,
        start_date,
        end_date
    )

    return render_template(
        "dashboard.html",
        license=license_data,
        companies=companies,
        selected_company=company,
        start_date=start_date,
        end_date=end_date,
        result=result,
        error=None
    )


@app.route("/activate", methods=["GET", "POST"])
def activate():

    # If a valid license already exists,
    # NEVER overwrite/change it.
    valid, license_data = get_license()

    if valid:
        return redirect(
            url_for("home")
        )

    message = ""

    if request.method == "POST":

        key = request.form.get(
            "license_key",
            ""
        ).strip()

        ok, data = verify_license(key)

        if ok:
            from license_util import save_license

            save_license(key)

            return redirect(
                url_for("home")
            )

        message = data

    return render_template(
        "activate.html",
        message=message
    )


@app.route("/chart/<path:filename>")
def chart(filename):

    path = (
        CHART_DIR
        / filename
    )

    if not path.exists():
        return "Chart not found.", 404

    return send_file(path)


if __name__ == "__main__":

    print("")
    print("==========================================")
    print(" STOCK MARKET ANALYSIS DASHBOARD")
    print("==========================================")
    print("Local URL:")
    print("http://127.0.0.1:5000")
    print("")
    print("License system: ACTIVE")
    print("Company selection: ENABLED")
    print("Date filtering: ENABLED")
    print("==========================================")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
