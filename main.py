from pathlib import Path

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.eda import run_eda
from src.indicators import add_moving_averages
from src.visualization import plot_close_and_moving_averages
from src.model import train_model
from src.evaluation import evaluate_model


BASE = Path(__file__).resolve().parent

DATA_FILE = (
    BASE
    / "data"
    / "raw"
    / "stock_data.csv"
)

CHART_FILE = (
    BASE
    / "outputs"
    / "charts"
    / "stock_price_ma.png"
)

PREDICTION_FILE = (
    BASE
    / "outputs"
    / "predictions"
    / "predictions.csv"
)

METRICS_FILE = (
    BASE
    / "outputs"
    / "metrics"
    / "model_metrics.txt"
)


def main():

    print("\n========================================")
    print(" STOCK MARKET ANALYSIS USING PYTHON")
    print(" AI & MACHINE LEARNING")
    print("========================================")

    if not DATA_FILE.exists():
        print("\nERROR: Dataset not found:")
        print(DATA_FILE)
        return

    print("\nLoading dataset...")
    df = load_data(DATA_FILE)

    print("Original dataset shape:", df.shape)

    print("\nCleaning dataset...")
    df = clean_data(df)

    print("Cleaned dataset shape:", df.shape)

    # EDA
    run_eda(df)

    # Moving averages
    print("\nCalculating moving averages...")
    df = add_moving_averages(df)

    # Chart
    print("\nCreating visualization...")
    plot_close_and_moving_averages(
        df,
        CHART_FILE
    )

    # Model
    print("\nTraining Linear Regression model...")

    model, X_test, y_test, predictions = train_model(df)

    # Evaluation
    metrics = evaluate_model(
        y_test,
        predictions
    )

    print("\n========================================")
    print(" MODEL EVALUATION")
    print("========================================")

    print("MAE :", metrics["MAE"])
    print("RMSE:", metrics["RMSE"])

    # Save predictions
    prediction_df = X_test.copy()

    prediction_df["actual_close"] = (
        y_test.values
    )

    prediction_df["predicted_close"] = (
        predictions
    )

    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False
    )

    # Save metrics
    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stock Market Analysis - Model Evaluation\n"
        )

        f.write(
            f"MAE: {metrics['MAE']}\n"
        )

        f.write(
            f"RMSE: {metrics['RMSE']}\n"
        )

    print(
        f"\nPredictions saved to:\n{PREDICTION_FILE}"
    )

    print(
        f"\nMetrics saved to:\n{METRICS_FILE}"
    )

    print("\n========================================")
    print(" PROJECT COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()
