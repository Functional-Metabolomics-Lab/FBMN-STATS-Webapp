import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import pingouin as pg


@st.cache_data(show_spinner="Testing for equal variance...")
def test_equal_variance(attribute, between, correction):
    # test for equal variance
    data = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    raw_pvals = []
    for f in st.session_state.data.columns:
        g0 = data.loc[data[attribute] == between[0], f]
        g1 = data.loc[data[attribute] == between[1], f]
        if g0.std() == 0 and g1.std() == 0:
            raw_pvals.append(np.nan)
        else:
            raw_pvals.append(stats.levene(g0, g1)[1])
    valid = [p for p in raw_pvals if not np.isnan(p)]
    corrected = pg.multicomp(valid, method=correction)[1] if valid else []
    it = iter(corrected)
    final = [next(it) if not np.isnan(p) else np.nan for p in raw_pvals]
    variance = pd.DataFrame({f"{between[0]} - {between[1]}": final})
    fig = px.histogram(
        variance,
        nbins=20,
        template="plotly_white",
        range_x=[-0.025, 1.025],
    )
    fig.update_layout(
        bargap=0.2,
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": f"TEST FOR EQUAL VARIANCE (LEVENE)", "font_color": "#3E3D53"},
        xaxis_title="p-value",
        yaxis_title="count",
        showlegend=False
    )
    return fig


@st.cache_data(show_spinner="Testing for equal variance (Bartlett)...")
def test_equal_variance_bartlett(attribute, between, correction):
    # test for equal variance using Bartlett's test
    data = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    raw_pvals = []
    for f in st.session_state.data.columns:
        g0 = data.loc[data[attribute] == between[0], f]
        g1 = data.loc[data[attribute] == between[1], f]
        if g0.std() == 0 or g1.std() == 0:
            raw_pvals.append(np.nan)
        else:
            raw_pvals.append(stats.bartlett(g0, g1)[1])
    valid = [p for p in raw_pvals if not np.isnan(p)]
    corrected = pg.multicomp(valid, method=correction)[1] if valid else []
    it = iter(corrected)
    final = [next(it) if not np.isnan(p) else np.nan for p in raw_pvals]
    variance = pd.DataFrame({f"{between[0]} - {between[1]}": final})
    fig = px.histogram(
        variance,
        nbins=20,
        template="plotly_white",
        range_x=[-0.025, 1.025],
    )
    fig.update_layout(
        bargap=0.2,
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": f"TEST FOR EQUAL VARIANCE (BARTLETT)", "font_color": "#3E3D53"},
        xaxis_title="p-value",
        yaxis_title="count",
        showlegend=False
    )
    return fig


@st.cache_data(show_spinner="Testing for normal distribution...")
def test_normal_distribution(attribute, between, correction):
    # test for normal distribution
    data = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    for b in between:
        if st.session_state.md[attribute].value_counts().loc[b] < 3:
            st.warning("You need at least 3 values in each option to test for normality!")
            return None
    normality_dict = {}
    for b in between:
        raw_pvals = []
        for f in st.session_state.data.columns:
            vals = data.loc[data[attribute] == b, f]
            if vals.std() == 0:
                raw_pvals.append(np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_pvals.append(stats.shapiro(vals)[1])
        valid = [p for p in raw_pvals if not np.isnan(p)]
        corrected = pg.multicomp(valid, method=correction)[1] if valid else []
        it = iter(corrected)
        normality_dict[f"{b}"] = [next(it) if not np.isnan(p) else np.nan for p in raw_pvals]
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
        xaxis_title="p-value",
        yaxis_title="count",
        showlegend=True
    )
    return fig



