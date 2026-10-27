"""Human-readable output: terminal summary, ROC plot, standalone HTML."""

from __future__ import annotations

import base64
import html
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audit import AuditReport

RISK_ADVICE = {
    "low": "No attack does meaningfully better than chance. Keep auditing as the model changes.",
    "moderate": "Some leakage. Usually fine for public data; review before training on anything sensitive.",
    "high": "An attacker can identify a meaningful share of training examples. "
    "Add regularization, early stopping, or train with differential privacy (DP-SGD).",
    "severe": "The model is memorizing its training set. Do not release it if the training data is sensitive.",
}

INK, MUTED, LINE, ACCENT = "#000000", "#888888", "#e6e6e6", "#3a4fbf"
GRAYS = ("#9a9aa3", "#cfcfd6")


def fit_metrics(report: AuditReport) -> tuple[str, str, str, str]:
    """(metric name, train, test, gap) formatted for display.

    Accuracy for classifiers; MSE for regressors, which has no fixed scale
    and so keeps significant digits rather than decimal places.
    """
    if report.task == "regression":
        return "mse", f"{report.train_mse:.4g}", f"{report.test_mse:.4g}", f"{report.generalization_gap:+.4g}"
    return "acc", f"{report.train_accuracy:.3f}", f"{report.test_accuracy:.3f}", f"{report.generalization_gap:+.3f}"


def text_summary(report: AuditReport) -> str:
    metric, train, test, gap = fit_metrics(report)
    lines = [
        f"trainleak audit: {report.model_name}",
        f"  members / non-members : {int(report.is_member.sum())} / {int((1 - report.is_member).sum())}",
        f"  train {metric} / test {metric}  : {train} / {test}  (gap {gap})",
        "",
        f"  {'attack':<18}{'AUC':>7}{'adv':>7}{'TPR@0.1%':>10}{'TPR@1%':>9}{'TPR@10%':>9}",
    ]
    for r in report.results.values():
        lines.append(
            f"  {r.name:<18}{r.auc:>7.3f}{r.advantage:>7.3f}"
            f"{r.tpr_at[0.001]:>10.3f}{r.tpr_at[0.01]:>9.3f}{r.tpr_at[0.1]:>9.3f}"
        )
    lines += ["", f"  risk: {report.risk.upper()}  (strongest attack: {report.strongest.name})"]
    lines.append(f"  {RISK_ADVICE[report.risk]}")
    return "\n".join(lines)


def style_log_roc_axes(ax, lo: float) -> None:
    """Log-log ROC axes in the house style, with ticks read as percentages."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ticks = [t for t in (0.001, 0.01, 0.1, 1.0) if t >= lo]
    labels = [f"{t:.1%}" if t < 0.01 else f"{t:.0%}" for t in ticks]
    ax.plot([lo, 1], [lo, 1], "--", color=MUTED, linewidth=1, label="Chance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Non-members wrongly flagged (FPR)", color=MUTED)
    ax.set_ylabel("Members found (TPR)", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)


def plot_roc(report: AuditReport, ax=None):
    """Log-log ROC curves, the view that exposes low-FPR leakage.

    The strongest attack is drawn in the accent color, the rest in gray.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(5.5, 5))
    strongest = report.strongest.name
    others = [r for r in report.results.values() if r.name != strongest]
    for i, r in enumerate(others):
        ax.plot(r.fpr, r.tpr, color=GRAYS[min(i, 1)], linewidth=1.4, label=f"{r.name}, AUC {r.auc:.2f}")
    s = report.results[strongest]
    ax.plot(s.fpr, s.tpr, color=ACCENT, linewidth=2, label=f"{s.name}, AUC {s.auc:.2f}")
    style_log_roc_axes(ax, lo=1 / (2 * report.is_member.sum()))
    ax.set_title(f"Membership inference: {report.model_name}", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return ax


def _roc_png_base64(report: AuditReport) -> str:
    # A bare Figure renders without pyplot, so writing a report never
    # switches the caller's matplotlib backend.
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.5, 5))
    plot_roc(report, ax=fig.add_subplot())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def write_html(report: AuditReport, path: str) -> None:
    metric, train, test, _ = fit_metrics(report)
    rows = "".join(
        f"<tr><td>{html.escape(r.name)}</td><td>{r.auc:.3f}</td><td>{r.advantage:.3f}</td>"
        f"<td>{r.tpr_at[0.001]:.3f}</td><td>{r.tpr_at[0.01]:.3f}</td><td>{r.tpr_at[0.1]:.3f}</td></tr>"
        for r in report.results.values()
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>trainleak report</title>
<style>
:root{{--ink:{INK};--muted:{MUTED};--line:{LINE};--accent:{ACCENT}}}
body{{font:15px/1.6 Inter,-apple-system,system-ui,sans-serif;max-width:680px;margin:48px auto;padding:0 16px;color:var(--ink);background:#fff}}
h1{{font-size:22px;font-weight:600;margin:0 0 4px}} .muted{{color:var(--muted)}} a{{color:var(--ink)}}
.risk{{color:var(--accent);font-weight:600;font-size:18px;margin:32px 0 4px}}
table{{border-collapse:collapse;width:100%;margin:32px 0;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}} th{{font-weight:600}}
th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%}}
</style></head><body>
<h1>Membership inference audit</h1>
<p class="muted">{html.escape(report.model_name)} &middot; {int(report.is_member.sum())} members vs
{int((1 - report.is_member).sum())} non-members &middot; train {metric} {train},
test {metric} {test}</p>
<p class="risk">Risk: {report.risk}</p>
<p>{html.escape(RISK_ADVICE[report.risk])}</p>
<table><thead><tr><th>Attack</th><th>AUC</th><th>Advantage</th><th>TPR@0.1%FPR</th><th>TPR@1%FPR</th><th>TPR@10%FPR</th></tr></thead>
<tbody>{rows}</tbody></table>
<img alt="ROC curves for each attack on log-log axes; the strongest attack is in blue" src="data:image/png;base64,{_roc_png_base64(report)}">
<p class="muted">The further a curve rises above the dashed line, the more the model leaks. TPR at low FPR counts
the training examples an attacker can name with confidence (Carlini et al., 2022).
Made with <a href="https://github.com/nancypatel12/trainleak">trainleak</a>.</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(page)
