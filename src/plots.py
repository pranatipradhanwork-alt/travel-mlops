"""Evaluation figures saved to reports/figures/ and logged to MLflow."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402


def _save(fig, name):
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def predicted_vs_actual(y_true, y_pred, title, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    idx = np.random.default_rng(0).choice(len(y_true), min(5000, len(y_true)), replace=False)
    yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
    axes[0].scatter(yt, yp, s=4, alpha=0.3)
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set(xlabel="Actual price", ylabel="Predicted price", title=title)
    axes[1].hist(yp - yt, bins=60)
    axes[1].set(xlabel="Residual (predicted - actual)", ylabel="Count", title="Residual distribution")
    return _save(fig, name)


def bar(labels, values, title, xlabel, name, color="#3b6fb6"):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.2))
    order = np.argsort(values)
    ax.barh(np.asarray(labels)[order], np.asarray(values)[order], color=color)
    ax.set(title=title, xlabel=xlabel)
    return _save(fig, name)


def grouped_bars(groups, series, title, ylabel, name):
    """series: {series_name: [value per group]}"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label)
    ax.set_xticks(x, groups, rotation=15)
    top = max(max(v) for v in series.values())
    ax.set_ylim(0, top * 1.2)  # headroom so the legend never covers bars
    ax.set(title=title, ylabel=ylabel)
    ax.legend(ncol=len(series), loc="upper left")
    return _save(fig, name)


def confusion(y_true, y_pred, labels, title, name):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, colorbar=False)
    ax.set_title(title)
    return _save(fig, name)
