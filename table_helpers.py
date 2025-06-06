from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, t
import pandas as pd
import numpy as np


def within_sig_holm(
    df: pd.DataFrame, measure: str, alpha: float = 0.05, higher_better: bool = True
) -> list:
    """
    Find the set of models that are within a certain significance level
    of the best model for a given measure.

    Parameters
    ----------
    df : DataFrame
        DataFrame with columns ['qid', 'name', 'measure', 'value']
    measure : str
        Which measure to compare
    alpha : float
        Significance level
    higher_better : bool
        If True, larger values are better; if False, smaller values are better.

    Returns
    -------
    within : list
        List of model names that are not significantly worse than the best.
    """
    # filter and pivot
    sub = df[df["measure"] == measure]
    mat = sub.pivot(index="qid", columns="name", values="value").dropna(axis=1)

    # choose best by mean
    means = mat.mean()
    best = means.idxmax() if higher_better else means.idxmin()

    # compare best to each other
    others = [m for m in mat.columns if m != best]
    pvals = []
    for m in others:
        t_stat, p_two = ttest_rel(mat[best], mat[m])
        # one-tailed p-value depending on direction
        if higher_better:
            p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        else:
            p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        pvals.append(p_one)

    # Holm–Bonferroni correction
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    # keep any that are NOT rejected (i.e., you can't say they're worse)
    within = [best] + [m for m, rej in zip(others, reject) if not rej]
    return within


def compute_ci(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    stats = (
        df.groupby(["name", "measure"])["value"]
        .agg(["mean", "std", "count"])
        .assign(
            se=lambda x: x["std"] / np.sqrt(x["count"]),
            ci=lambda x: x["se"] * t.ppf(1 - alpha / 2, x["count"] - 1),
        )
        .reset_index()
    )
    return stats[["name", "measure", "mean", "ci"]]


def format_and_style(
    results: pd.DataFrame,
    preferences: dict = {},
    decimals: int = 3,
    alpha: float = 0.05,
    measures_order: list = None,
):
    """
    Formats results into a styled DataFrame, highlighting models within
    significance of the best and bolding the best per measure.

    Parameters
    ----------
    results : DataFrame
        Long-form results with columns ['qid','name','measure','value']
    preferences : dict
        Mapping from measure -> bool (True if higher is better).
    decimals : int
        Number of decimal places for display.
    alpha : float
        Significance level for Holm correction.
    measures_order : list, optional
        Desired ordering of measures in the display.

    Returns
    -------
    Styler
    """
    # compute statistics
    ci_df = compute_ci(results, alpha)
    # determine measure list and order
    if measures_order is not None:
        measures = measures_order
    else:
        measures = ci_df["measure"].unique().tolist()

    # pivot for display
    pivot = ci_df.pivot(index="name", columns="measure")

    # determine best and significance groups per measure
    best_per_measure = {}
    sig_groups = {}
    for m in measures:
        hb = preferences.get(m, True)
        best_per_measure[m] = (
            pivot[("mean", m)].idxmax() if hb else pivot[("mean", m)].idxmin()
        )
        sig_groups[m] = within_sig_holm(results, m, alpha=alpha, higher_better=hb)

    # build clean display
    clean = pd.DataFrame(index=pivot.index)
    for m in measures:
        means = pivot[("mean", m)]
        cis = pivot[("ci", m)]
        clean[m] = [
            f"{mn:.{decimals}f} \u00b1 {ci:.{decimals}f}" for mn, ci in zip(means, cis)
        ]

    # styling
    def style_row(row):
        styles = []
        for m in measures:
            css = ""
            if row.name in sig_groups[m]:
                css += "background-color: lightgrey; color: black;"
            if row.name == best_per_measure[m]:
                css += "font-weight: bold;"
            styles.append(css)
        return styles

    styled = clean.style.apply(style_row, axis=1)
    return styled


def build_latex_table(
    results: pd.DataFrame,
    alpha: float = 0.05,
    preferences: dict = {},
    decimals: int = 3,
    measures_order: list = None,
) -> str:
    """
    Build a LaTeX table of summary statistics with significance annotations,
    allowing custom direction preferences and measures ordering.

    Parameters
    ----------
    results : DataFrame
        Long-form results with columns ['qid','name','measure','value']
    alpha : float
        Significance level for Holm correction.
    preferences : dict
        Mapping from measure -> bool (True if higher is better).
    decimals : int
        Number of decimal places in the LaTeX output.
    measures_order : list, optional
        Desired ordering of measures in the table.

    Returns
    -------
    wrapper : str
        A string containing the LaTeX code for the table.
    """
    # 1) compute CI
    ci = compute_ci(results, alpha)
    # determine measures and order
    unique_measures = ci["measure"].unique().tolist()
    measures = measures_order if measures_order is not None else unique_measures

    # 2) determine significance groups and best per measure
    sig = {}
    best_per_measure = {}
    pivot_ci = ci.pivot(index="name", columns="measure")
    for m in measures:
        hb = preferences.get(m, True)
        sig[m] = within_sig_holm(results, m, alpha=alpha, higher_better=hb)
        if hb:
            best_per_measure[m] = pivot_ci[("mean", m)].idxmax()
        else:
            best_per_measure[m] = pivot_ci[("mean", m)].idxmin()

    # 3) build rows with LaTeX markup
    rows = []
    for name in pivot_ci.index:
        row = [name.replace("_", "\\_")]
        for m in measures:
            mn = pivot_ci.loc[name, ("mean", m)]
            ci_val = pivot_ci.loc[name, ("ci", m)]
            core = f"{mn:.{decimals}f}\\pm{ci_val:.{decimals}f}"
            if name == best_per_measure[m]:
                entry = f"$\\mathbf{{{core}}}$"
            else:
                entry = f"${core}$"
            if name in sig[m]:
                entry = f"\\nsig{{{entry}}}"
            row.append(entry)
        rows.append(row)

    # 4) assemble DataFrame
    cols = ["Name"] + [
        f"{m.replace('_', '\\_')}"
        + (" $\\uparrow$" if preferences.get(m, True) else " $\\downarrow$")
        for m in measures
    ]
    latex_df = pd.DataFrame(rows, columns=cols)

    # 5) export to LaTeX
    col_fmt = "l " + " ".join("c" for _ in measures)
    body = latex_df.to_latex(
        index=False,
        escape=False,
        column_format=col_fmt,
        bold_rows=False,
        longtable=False,
        header=True,
    )

    wrapper = (
        "\\begin{table*}[h]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n" + body + "}\n"
        "\\caption{…}\n"
        "\\label{tab:…}\n"
        "\\end{table*}\n"
    )
    return wrapper
