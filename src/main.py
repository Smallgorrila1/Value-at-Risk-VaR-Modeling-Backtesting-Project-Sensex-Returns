import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pyextremes import get_model


def historical_var(returns, alpha=0.05):
    return np.percentile(returns, 100 * alpha)


def ewma_var(returns, lam=0.94, alpha=0.05):
    ewma_var = (returns ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    sigma_t = np.sqrt(ewma_var)
    sigma_T = sigma_t.iloc[-1]
    adj_returns = returns * (sigma_T / sigma_t.replace(0, 1e-12))
    return np.percentile(adj_returns.dropna(), 100 * alpha)


def parametric_var(returns, alpha=0.05):
    mu_g = np.mean(returns)
    sigma_g = np.std(returns, ddof=1)
    z_alpha = norm.ppf(alpha)
    var = 1 - np.exp(mu_g - z_alpha * sigma_g)
    return var


def pot_var(returns, threshold=0.008, confidence=0.99):
    losses = -returns
    extremes_series = losses[losses > threshold]

    if len(extremes_series) < 10:
        raise ValueError("Too few exceedances above threshold; try lowering threshold.")

    model = get_model("MLE", extremes=extremes_series, distribution="genpareto")
    shape = model.fit_parameters["c"]
    scale = model.fit_parameters["scale"]

    n = len(losses)
    n_u = len(extremes_series)
    exceedance_prob = n / n_u * (1 - confidence)

    VaR_theory = threshold + (scale / shape) * ((exceedance_prob ** (-shape)) - 1)
    VaR_model = model.get_return_value(exceedance_prob)
    return VaR_theory * -1, VaR_model * -1, shape, scale


def run_all_var(returns, year, graph=True):
    var_95_hist = historical_var(returns, 0.05)
    var_99_hist = historical_var(returns, 0.01)
    var_95_ewma = ewma_var(returns, lam=0.94, alpha=0.05)
    var_99_ewma = ewma_var(returns, lam=0.94, alpha=0.01)
    var_95_evt, _, shape, scale = pot_var(returns, threshold=0.008, confidence=0.95)
    var_99_evt, _, _, _ = pot_var(returns, threshold=0.008, confidence=0.99)
    var_95_para = parametric_var(returns)
    var_99_para = parametric_var(returns, alpha=0.01)

    results = pd.DataFrame({
        "Model": [
            "Historical",
            "EWMA (Vol-Adj)",
            "Parametric",
            "EVT-POT"
        ],
        "VaR 95%": [
            var_95_hist,
            var_95_ewma,
            var_95_para,
            var_95_evt
        ],
        "VaR 99%": [
            var_99_hist,
            var_99_ewma,
            var_99_para,
            var_99_evt
        ],
    })

    print(f"\n📊 Value-at-Risk Comparison Table ({year})")
    print(results.to_string(index=False))

    if graph:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["blue", "green", "orange", "red"]
        tail_cut_95 = np.percentile(returns, 100)
        returns_95_tail = returns[returns <= tail_cut_95]

        axes[0].hist(returns, bins=20, density=True, color="lightgrey", alpha=0.9,
                     edgecolor='grey')  # Increased bins for detail
        axes[0].set_xlim(returns.min() * 1.1, 0)  # Adjust x-lim to show full tail if needed
        axes[0].set_title("Left-Tail Distribution (95% VaR)")
        axes[0].set_xlabel("Returns")
        axes[0].set_ylabel("Density")

        for i, model in enumerate(results["Model"]):
            var_value = results.loc[i, "VaR 95%"]
            axes[0].axvline(var_value, color=colors[i], linestyle="--", linewidth=2,
                            label=f"{model}: {var_value:.4f}")

        axes[0].legend()

        # --- 99% VaR plot ---
        axes[1].hist(returns, bins=20, density=True, color="lightgrey", alpha=0.9, edgecolor='grey')  # Increased # bins for detail
        axes[1].set_xlim(returns.min() * 1.1, 0)  # Adjust x-lim to show full tail if needed
        axes[1].set_title("Left-Tail Distribution (99% VaR)")
        axes[1].set_xlabel("Returns")

        for i, model in enumerate(results["Model"]):
            var_value = results.loc[i, "VaR 99%"]
            axes[1].axvline(var_value, color=colors[i], linestyle="--", linewidth=2,
                            label=f"{model}: {var_value:.4f}")

        axes[1].legend()

        plt.suptitle(f"Left-Tail VaR Comparison — {year}", fontsize=14, weight='bold')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to prevent suptitle overlap
        plt.show()
    return results


def backtest_var(returns, var_series, alpha=0.05, model_name="Model", plot=True, results=True):
    if not isinstance(var_series, pd.Series):
        var_series = pd.Series([var_series] * len(returns), index=returns.index)

    exceptions = (returns < var_series).astype(int)
    n = len(exceptions)
    x = exceptions.sum()
    p = alpha

    if x == 0 or x == n:
        LR_uc = 0
    else:
        LR_uc = -2 * np.log(((1 - p) ** (n - x) * p ** x) /
                            ((1 - x / n) ** (n - x) * (x / n) ** x))

    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = exceptions.iloc[i - 1], exceptions.iloc[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x): return np.log(x) if x > 0 else 0
    LR_ind = -2 * (safe_log(((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
                            ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)))
    LR_cc = LR_uc + LR_ind

    if results:
        print(f"\nBacktesting Results — {model_name}")
        print(f"Confidence Level: {(1 - alpha) * 100:.0f}%")
        print(f"Observations: {n} | Exceptions: {x} | Failure Rate: {x/n:.2%}")
        print(f"LR_uc (Unconditional): {LR_uc:.4f}")
        print(f"LR_ind (Independence): {LR_ind:.4f}")
        print(f"LR_cc (Conditional): {LR_cc:.4f}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(returns.index, returns, label="Returns", alpha=0.6)
        plt.plot(var_series.index, var_series, color="red", label=f"VaR {int((1-alpha)*100)}%")
        plt.scatter(returns[exceptions == 1].index, returns[exceptions == 1],
                    color="black", label="Exceptions", s=25)
        plt.title(f"VaR Backtesting - {model_name}")
        plt.legend()
        plt.show()

    return {
        "Model": model_name,
        "Alpha": alpha,
        "Exceptions": x,
        "Failure Rate": x / n,
        "LR_uc": LR_uc,
        "LR_ind": LR_ind,
        "LR_cc": LR_cc
    }


def run_backtests(returns, backtest_returns):
    var_hist_95 = historical_var(returns, 0.05)
    var_ewma_95 = ewma_var(returns, lam=0.94, alpha=0.05)
    var_para_95 = parametric_var(returns, .05)
    var_evt_95, _, _, _ = pot_var(returns, threshold=0.008, confidence=0.95)
    var_hist_99 = historical_var(returns, 0.01)
    var_ewma_99 = ewma_var(returns, lam=0.94, alpha=0.01)
    var_para_99 = parametric_var(returns, .01)
    var_evt_99, _, _, _ = pot_var(returns, threshold=0.008, confidence=0.99)

    results = []
    results.append(backtest_var(backtest_returns, var_hist_95, alpha=0.05, model_name="Historical VaR 95%", results=False))
    results.append(backtest_var(backtest_returns, var_ewma_95, alpha=0.05, model_name="EWMA VaR 95%", results=False))
    results.append(backtest_var(backtest_returns, var_para_95, alpha=.05, model_name='Parametric VaR 95%', results=False))
    results.append(backtest_var(backtest_returns, var_evt_95, alpha=0.05, model_name="EVT-POT VaR 95%", results=False))
    results.append(backtest_var(backtest_returns, var_hist_99, alpha=0.01, model_name="Historical VaR 99%", results=False))
    results.append(backtest_var(backtest_returns, var_ewma_99, alpha=0.01, model_name="EWMA VaR 99%", results=False))
    results.append(backtest_var(backtest_returns, var_para_99, alpha=.01, model_name='Parametric VaR 99%', results=False))
    results.append(backtest_var(backtest_returns, var_evt_99, alpha=0.01, model_name="EVT-POT VaR 99%", results=False))

    backtest_summary = pd.DataFrame(results)
    print("\nBacktesting Summary Table")
    print(backtest_summary.to_string(index=False))
    return backtest_summary


sheet_name = "Returns SENSEX"
file_path = "sensex_weight.xlsx"
year = 2023

df = pd.read_excel(file_path, sheet_name=sheet_name)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

returns = df.loc[f"{year}-01-01":f"{year}-12-31", "Sensex Returns"]
backtest_returns = df.loc[f"{year+1}-01-01":f"{year+1}-12-31", "Sensex Returns"]

summary_table_2023 = run_all_var(returns, year=year)
summary_table_2024 = run_all_var(backtest_returns, 2024, graph=False)
run_backtests(returns,backtest_returns)
summary_table_2023.to_excel("VaR_Results_2023.xlsx", index=False)
run_backtests(returns,backtest_returns).to_excel("VaR_Backtest_Summary.xlsx", index=False)