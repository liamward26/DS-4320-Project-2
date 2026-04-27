"""
04_visualization.py

Creates publication-quality Plotly visualizations from readmission analysis outputs.
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging() -> None:
    """Configure logging to both a file and the console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_DIR / "visualization.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load analysis output CSV files."""
    return (
        pd.read_csv(OUTPUT_DIR / "model_results.csv"),
        pd.read_csv(OUTPUT_DIR / "feature_importance.csv"),
        pd.read_csv(OUTPUT_DIR / "analysis_dataset.csv"),
    )


def clean_feature_name(feature: str) -> str:
    """Clean sklearn feature names for chart labels."""
    return (
        feature.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
        .title()
    )


def save_plot(fig, filename: str) -> None:
    """Save a Plotly figure as both HTML and PNG."""
    html_path = FIGURE_DIR / f"{filename}.html"
    png_path = FIGURE_DIR / f"{filename}.png"

    fig.write_html(html_path)
    fig.write_image(png_path, scale=3)

    logging.info("Saved %s and %s", html_path, png_path)


def plot_model_comparison(model_results: pd.DataFrame) -> None:
    """Create a grouped bar chart comparing model performance metrics."""
    metrics = ["accuracy", "precision", "recall", "roc_auc"]

    plot_df = model_results.melt(
        id_vars="model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    plot_df["Metric"] = plot_df["Metric"].str.replace("_", "-").str.title()

    fig = px.bar(
        plot_df,
        x="Metric",
        y="Score",
        color="model",
        barmode="group",
        title="Model Performance for 30-Day Readmission Prediction",
        text=plot_df["Score"].round(2),
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="Score",
        xaxis_title="Evaluation Metric",
        legend_title="Model",
        yaxis=dict(range=[0, 1]),
        font=dict(size=14),
        title_font=dict(size=20),
    )

    fig.update_traces(textposition="outside")

    save_plot(fig, "model_comparison")


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Create a horizontal bar chart of the top readmission predictors."""
    top_features = feature_importance.head(15).copy()
    top_features["feature"] = top_features["feature"].apply(clean_feature_name)
    top_features = top_features.sort_values("importance", ascending=True)

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Predictors of 30-Day Hospital Readmission",
        text=top_features["importance"].round(3),
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Random Forest Feature Importance",
        yaxis_title="Feature",
        font=dict(size=14),
        title_font=dict(size=20),
        height=650,
    )

    fig.update_traces(textposition="outside")

    save_plot(fig, "feature_importance")


def plot_readmission_by_prior_encounters(analysis_dataset: pd.DataFrame) -> None:
    """Create a chart showing readmission rate by prior encounter group."""
    df = analysis_dataset.copy()

    df["prior_encounter_group"] = pd.cut(
        df["prior_encounters"],
        bins=[-1, 0, 2, 5, 10, float("inf")],
        labels=["0", "1-2", "3-5", "6-10", "11+"],
    )

    readmission_rates = (
        df.groupby("prior_encounter_group", observed=False)["readmitted_30_days"]
        .mean()
        .reset_index()
    )

    readmission_rates["readmission_percent"] = (
        readmission_rates["readmitted_30_days"] * 100
    )

    fig = px.bar(
        readmission_rates,
        x="prior_encounter_group",
        y="readmission_percent",
        title="30-Day Readmission Rate by Prior Encounter History",
        text=readmission_rates["readmission_percent"].round(1),
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Number of Prior Encounters",
        yaxis_title="Readmission Rate (%)",
        font=dict(size=14),
        title_font=dict(size=20),
    )

    fig.update_traces(texttemplate="%{text}%", textposition="outside")

    save_plot(fig, "readmission_by_prior_encounters")


def run_visualization() -> None:
    """Run the full visualization pipeline."""
    setup_logging()
    logging.info("Starting Plotly visualization pipeline")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model_results, feature_importance, analysis_dataset = load_outputs()

    plot_model_comparison(model_results)
    plot_feature_importance(feature_importance)
    plot_readmission_by_prior_encounters(analysis_dataset)

    logging.info("Visualization pipeline complete")
    print(f"Saved Plotly figures to {FIGURE_DIR}")


if __name__ == "__main__":
    run_visualization()