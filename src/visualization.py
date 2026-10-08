import matplotlib.pyplot as plt

def plot_close_and_moving_averages(df, output_path):
    plt.figure(figsize=(14, 7))

    plt.plot(
        df["date"],
        df["close"],
        label="Close Price",
        linewidth=1.2
    )

    plt.plot(
        df["date"],
        df["ma20"],
        label="20-Day Moving Average",
        linewidth=1.5
    )

    plt.plot(
        df["date"],
        df["ma50"],
        label="50-Day Moving Average",
        linewidth=1.5
    )

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(
        "Stock Price Trend with Moving Averages"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nChart saved to: {output_path}")
