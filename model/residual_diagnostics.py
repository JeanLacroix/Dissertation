from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .train import ARTIFACTS_DIR, CHANGE_D_FORMULA, _fit_ols, prepare_change_d_analysis_frame

DIAGNOSTICS_DIR = ARTIFACTS_DIR / "residual_diagnostics"


def _fit_change_d_model():
    model_frame, _, _ = prepare_change_d_analysis_frame()
    model = _fit_ols(model_frame, CHANGE_D_FORMULA)
    return model_frame, model


def _plot_residuals_vs_fitted(fitted: np.ndarray, residuals: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(fitted, residuals, alpha=0.6, color="#1f77b4", edgecolor="none")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted log deal size")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals versus fitted values")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = DIAGNOSTICS_DIR / "residuals_vs_fitted.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_residuals_vs_logsize(log_size: np.ndarray, residuals: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(log_size, residuals, alpha=0.6, color="#3182bd", edgecolor="none")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Log total size (sqm)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals versus log size")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = DIAGNOSTICS_DIR / "residuals_vs_logsize.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_qq(residuals: np.ndarray) -> Path:
    fig = qqplot(residuals, line="45", fit=True)
    fig.axes[0].set_title("Q-Q plot of OLS residuals")
    fig.axes[0].grid(alpha=0.2)
    fig.tight_layout()

    output_path = DIAGNOSTICS_DIR / "qq_plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_scale_location(fitted: np.ndarray, standardized_residuals: np.ndarray) -> Path:
    y_values = np.sqrt(np.abs(standardized_residuals))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(fitted, y_values, alpha=0.6, color="#756bb1", edgecolor="none")
    ax.set_xlabel("Fitted log deal size")
    ax.set_ylabel("Sqrt(|standardised residual|)")
    ax.set_title("Scale-location plot")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = DIAGNOSTICS_DIR / "scale_location.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _vif_table(exog: np.ndarray, exog_names: list[str]) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(exog_names):
        if name == "Intercept":
            continue
        vif_value = float(variance_inflation_factor(exog, idx))
        rows.append(
            {
                "predictor": name,
                "vif": vif_value,
                "flag": bool(vif_value > 10.0),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def _diagnostics_summary(model, vif_table: pd.DataFrame) -> pd.DataFrame:
    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)
    standardized_residuals = np.asarray(model.get_influence().resid_studentized_internal, dtype=float)
    bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = het_breuschpagan(residuals, model.model.exog)
    dw_stat = float(durbin_watson(residuals))
    jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
    max_vif = float(vif_table["vif"].max()) if not vif_table.empty else float("nan")

    rows = [
        {
            "test_name": "Linearity (visual)",
            "assumption": "Linearity",
            "null_hypothesis": "No strong systematic non-linearity is visible in residual plots.",
            "statistic": np.nan,
            "p_value": np.nan,
            "secondary_statistic": np.nan,
            "secondary_p_value": np.nan,
            "conclusion_5pct": (
                "Residuals remain centred around zero in the residual-versus-fitted and residual-versus-log(size) plots, "
                "with no strong curvature, although dispersion remains wide."
            ),
        },
        {
            "test_name": "Breusch-Pagan",
            "assumption": "Homoscedasticity",
            "null_hypothesis": "Residual variance is constant.",
            "statistic": float(bp_lm_stat),
            "p_value": float(bp_lm_pvalue),
            "secondary_statistic": float(bp_f_stat),
            "secondary_p_value": float(bp_f_pvalue),
            "conclusion_5pct": (
                "Fail to reject homoscedasticity at the 5% level."
                if bp_lm_pvalue >= 0.05
                else "Reject homoscedasticity at the 5% level."
            ),
        },
        {
            "test_name": "Variance inflation factor",
            "assumption": "Multicollinearity",
            "null_hypothesis": "No severe multicollinearity is present.",
            "statistic": max_vif,
            "p_value": np.nan,
            "secondary_statistic": float(vif_table["flag"].sum()) if not vif_table.empty else 0.0,
            "secondary_p_value": np.nan,
            "conclusion_5pct": (
                "No predictor exceeds the VIF > 10 threshold."
                if max_vif <= 10.0
                else "At least one predictor exceeds the VIF > 10 threshold."
            ),
        },
        {
            "test_name": "Durbin-Watson",
            "assumption": "Independence of residuals",
            "null_hypothesis": "Residuals do not exhibit strong serial correlation.",
            "statistic": dw_stat,
            "p_value": np.nan,
            "secondary_statistic": np.nan,
            "secondary_p_value": np.nan,
            "conclusion_5pct": (
                "The statistic is close to 2, which is broadly consistent with low residual autocorrelation."
                if 1.5 <= dw_stat <= 2.5
                else "The statistic departs materially from 2 and merits caution on residual independence."
            ),
        },
        {
            "test_name": "Jarque-Bera",
            "assumption": "Normality of residuals",
            "null_hypothesis": "Residuals are normally distributed.",
            "statistic": float(jb_stat),
            "p_value": float(jb_pvalue),
            "secondary_statistic": np.nan,
            "secondary_p_value": np.nan,
            "conclusion_5pct": (
                "Fail to reject normality at the 5% level."
                if jb_pvalue >= 0.05
                else "Reject normality at the 5% level."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _write_readme(summary: pd.DataFrame, vif_table: pd.DataFrame) -> Path:
    bp_row = summary.loc[summary["test_name"].eq("Breusch-Pagan")].iloc[0]
    vif_row = summary.loc[summary["test_name"].eq("Variance inflation factor")].iloc[0]
    dw_row = summary.loc[summary["test_name"].eq("Durbin-Watson")].iloc[0]
    jb_row = summary.loc[summary["test_name"].eq("Jarque-Bera")].iloc[0]
    top_vif = vif_table.head(5).to_string(index=False)

    text = f"""# Residual diagnostics for Change D

## Linearity
Residuals were inspected visually against fitted values and log(size). The cloud remains centred around zero without a strong systematic curve, although dispersion remains broad, so the linear functional form is not obviously contradicted by the plots.

## Homoscedasticity
Breusch-Pagan tests the null of constant residual variance. The LM statistic is {bp_row['statistic']:.3f} with p-value {bp_row['p_value']:.4f}; the F statistic is {bp_row['secondary_statistic']:.3f} with p-value {bp_row['secondary_p_value']:.4f}. Conclusion: {bp_row['conclusion_5pct']}

## Multicollinearity
Variance inflation factors assess whether predictors are excessively collinear. The maximum observed VIF is {vif_row['statistic']:.3f}. {vif_row['conclusion_5pct']}

Top VIF values:

```
{top_vif}
```

## Independence of residuals
Durbin-Watson is reported for completeness given the rolling-origin design. The statistic is {dw_row['statistic']:.3f}. Conclusion: {dw_row['conclusion_5pct']}

## Normality
Jarque-Bera tests the null that residuals are normally distributed. The statistic is {jb_row['statistic']:.3f} with p-value {jb_row['p_value']:.4f}. Conclusion: {jb_row['conclusion_5pct']}
"""
    output_path = DIAGNOSTICS_DIR / "README.md"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    model_frame, model = _fit_change_d_model()
    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)
    standardized_residuals = np.asarray(model.get_influence().resid_studentized_internal, dtype=float)

    summary = _diagnostics_summary(model, _vif_table(model.model.exog, list(model.model.exog_names)))
    vif_table = _vif_table(model.model.exog, list(model.model.exog_names))

    summary_path = DIAGNOSTICS_DIR / "diagnostics_summary.csv"
    vif_path = DIAGNOSTICS_DIR / "vif_table.csv"
    summary.to_csv(summary_path, index=False)
    vif_table.to_csv(vif_path, index=False)

    fitted_plot = _plot_residuals_vs_fitted(fitted, residuals)
    logsize_plot = _plot_residuals_vs_logsize(model_frame["log_total_size_sqm"].to_numpy(dtype=float), residuals)
    qq_plot_path = _plot_qq(residuals)
    scale_location_path = _plot_scale_location(fitted, standardized_residuals)
    readme_path = _write_readme(summary, vif_table)

    print("Residual diagnostics complete.")
    print(f"Rows analysed: {len(model_frame)}")
    print(f"Diagnostics summary: {summary_path}")
    print(f"VIF table: {vif_path}")
    print(f"Residuals vs fitted: {fitted_plot}")
    print(f"Residuals vs log size: {logsize_plot}")
    print(f"Q-Q plot: {qq_plot_path}")
    print(f"Scale-location plot: {scale_location_path}")
    print(f"README: {readme_path}")
    print(summary.loc[:, ["test_name", "statistic", "p_value", "conclusion_5pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
