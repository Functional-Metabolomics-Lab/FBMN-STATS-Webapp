import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import pingouin as pg

# As in the FBMN-STATS protocol (Step 58), the per-feature p-values of the assumption tests are
# corrected for multiple testing with the method selected in the sidebar (Benjamini-Hochberg by default).
# The data and metadata are passed as arguments so the cache is invalidated when data
# preparation is redone.


def _correct(raw_pvals, correction):
    raw = np.asarray(raw_pvals, dtype=float)
    out = np.full(raw.shape, np.nan)
    valid = ~np.isnan(raw)
    if valid.any():
        out[valid] = pg.multicomp(raw[valid], method=correction)[1]
    return list(out)


def _p_label(correction):
    return "p-value" if correction in (None, "none") else f"p-value (corrected: {correction})"


def _variance_histogram(pvals, between, title, correction):
    variance = pd.DataFrame({f"{between[0]} - {between[1]}": pvals})
    fig = px.histogram(
        variance,
        nbins=20,
        template="plotly_white",
        range_x=[-0.025, 1.025],
    )
    fig.update_layout(
        bargap=0.2,
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        xaxis_title=_p_label(correction),
        yaxis_title="count",
        showlegend=False
    )
    return fig


@st.cache_data(show_spinner="Testing for equal variance...")
def test_equal_variance(data, md, attribute, between, correction):
    # test for equal variance (scipy's default center='median', i.e. the Brown-Forsythe variant of Levene's test)
    df = pd.concat([data, md[[attribute]]], axis=1)
    raw_pvals = []
    for f in data.columns:
        g0 = df.loc[df[attribute] == between[0], f].dropna()
        g1 = df.loc[df[attribute] == between[1], f].dropna()
        if len(g0) < 2 or len(g1) < 2 or (g0.std() == 0 and g1.std() == 0):
            raw_pvals.append(np.nan)
        else:
            raw_pvals.append(stats.levene(g0, g1)[1])
    return _variance_histogram(_correct(raw_pvals, correction), between, "TEST FOR EQUAL VARIANCE (LEVENE)", correction)


@st.cache_data(show_spinner="Testing for equal variance (Bartlett)...")
def test_equal_variance_bartlett(data, md, attribute, between, correction):
    # test for equal variance using Bartlett's test
    df = pd.concat([data, md[[attribute]]], axis=1)
    raw_pvals = []
    for f in data.columns:
        g0 = df.loc[df[attribute] == between[0], f].dropna()
        g1 = df.loc[df[attribute] == between[1], f].dropna()
        if len(g0) < 2 or len(g1) < 2 or g0.std() == 0 or g1.std() == 0:
            raw_pvals.append(np.nan)
        else:
            raw_pvals.append(stats.bartlett(g0, g1)[1])
    return _variance_histogram(_correct(raw_pvals, correction), between, "TEST FOR EQUAL VARIANCE (BARTLETT)", correction)


@st.cache_data(show_spinner="Testing for normal distribution...")
def test_normal_distribution(data, md, attribute, between, correction):
    # test for normal distribution
    df = pd.concat([data, md[[attribute]]], axis=1)
    for b in between:
        if md[attribute].value_counts().get(b, 0) < 3:
            st.warning("You need at least 3 values in each option to test for normality!")
            return None
    normality_dict = {}
    for b in between:
        raw_pvals = []
        for f in data.columns:
            vals = df.loc[df[attribute] == b, f].dropna()
            if len(vals) < 3 or vals.std() == 0:
                raw_pvals.append(np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_pvals.append(stats.shapiro(vals)[1])
        normality_dict[f"{b}"] = _correct(raw_pvals, correction)
    normality = pd.DataFrame(normality_dict)

    fig = px.histogram(
        normality,
        nbins=20,
        template="plotly_white",
        range_x=[-0.025, 1.025],
        barmode="group",
    )

    fig.update_layout(
        bargap=0.2,
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": f"TEST FOR NORMALITY (SHAPIRO-WILK)", "font_color": "#3E3D53"},
        xaxis_title=_p_label(correction),
        yaxis_title="count",
        showlegend=True
    )
    return fig
