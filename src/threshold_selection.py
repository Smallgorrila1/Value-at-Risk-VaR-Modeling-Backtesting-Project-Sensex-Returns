from pyextremes import plot_mean_residual_life, plot_parameter_stability, plot_threshold_stability
import pandas as pd
import matplotlib.pyplot as plt


def main():
    returns_df = pd.read_excel("sensex_weight.xlsx", sheet_name="Returns SENSEX")

    returns_df["Date"] = pd.to_datetime(returns_df["Date"])
    returns_df.set_index("Date", inplace=True)

    train_returns = returns_df.loc["2023-01-01":"2023-12-31"]["Sensex Returns"]
    losses = train_returns * -1
    print(losses)

    plot_mean_residual_life(losses)
    plot_parameter_stability(losses)
    plt.show()


if __name__ == '__main__':
    main()