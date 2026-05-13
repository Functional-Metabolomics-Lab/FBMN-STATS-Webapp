import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import friedmanchisquare
import time
from src.utils import get_feature_name_map


# ---------------------------------------------------------------------------
# Friedman omnibus test
# ---------------------------------------------------------------------------

def gen_friedman_data(group_data, _progress_callback=None):
    """Yield (metabolite, p, chi2) for each metabolite column
    across 3+ paired groups (equal-length arrays, matched by row index)."""
    total = len(group_data[0].columns)
    start_time = time.time()
    for idx, col in enumerate(group_data[0].columns):
        arrays = [df[col].values for df in group_data]
        try:
            statistic, p = friedmanchisquare(*arrays)
        except ValueError:
            if _progress_callback is not None:
                elapsed = time.time() - start_time
                done = idx + 1
                est_left = (elapsed / done) * (total - done) if done > 0 else 0
                _progress_callback(done, total, est_left)
            continue

        if _progress_callback is not None:
            elapsed = time.time() - start_time
            done = idx + 1
            est_left = (elapsed / done) * (total - done) if done > 0 else 0
            _progress_callback(done, total, est_left)

        yield col, p, statistic


def add_p_correction_to_friedman(df, correction):
    if "p-corrected" not in df.columns:
        df.insert(2, "p-corrected",
                  pg.multicomp(df["p"].astype(float), method=correction)[1])
    if "significant" not in df.columns:
        df.insert(3, "significant", df["p-corrected"] < 0.05)
    df.sort_values("p", inplace=True)
    return df


def friedman_test(attribute, correction, elements, _progress_callback=None):
    """Run a Friedman test per metabolite for *elements* groups within *attribute*.

    Paired design: the groups are truncated to the shortest group length so that
    row *i* in each group represents the same subject / matched observation.
    """
    combined = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    if elements is not None:
        combined = combined[combined[attribute].isin(elements)]

    groups = sorted(combined[attribute].dropna().unique())
    if not groups:
        st.session_state.friedman_attempted_metabolites = 0
        st.session_state.friedman_returned_metabolites = 0
        return pd.DataFrame()

    metabolite_cols = []
    for col in st.session_state.data.columns:
        valid_for_all = True
        for g in groups:
            n_nonnull = combined.loc[combined[attribute] == g, col].dropna().shape[0]
            if n_nonnull < 2:
                valid_for_all = False
                break
        if valid_for_all:
            metabolite_cols.append(col)
    st.session_state.friedman_attempted_metabolites = len(metabolite_cols)

    if not metabolite_cols:
        st.session_state.friedman_returned_metabolites = 0
        return pd.DataFrame()

    # Build per-group dataframes, reset index for row-wise pairing
    group_data_raw = [
        combined[combined[attribute] == g].loc[:, metabolite_cols].reset_index(drop=True)
        for g in groups
    ]

    # Truncate to equal length (shortest group)
    min_len = min(len(gdf) for gdf in group_data_raw)
    if min_len < 2:
        st.session_state.friedman_returned_metabolites = 0
        return pd.DataFrame()
    group_data = [gdf.iloc[:min_len] for gdf in group_data_raw]

    df = pd.DataFrame(
        np.fromiter(
            gen_friedman_data(group_data, _progress_callback=_progress_callback),
            dtype=[("metabolite", "U100"), ("p", "f"), ("statistic", "f")],
        )
    )
    if df.empty:
        st.session_state.friedman_returned_metabolites = 0
        return df
    df = df.dropna()
    df = add_p_correction_to_friedman(df, correction)
    df = df[df["metabolite"] != attribute]
    st.session_state.friedman_returned_metabolites = len(df)
    return df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


@st.cache_resource
def get_friedman_plot(friedman_df, color_by=None):
    friedman_clean = friedman_df[friedman_df["metabolite"].notna()].copy()

    if "significant" in friedman_clean.columns:
        friedman_clean["significant"] = friedman_clean["significant"].fillna(False).astype(bool)
    else:
        friedman_clean["significant"] = False

    unique_metabolites = friedman_clean["metabolite"].unique()
    total_points = len(unique_metabolites)
    n_significant = int(friedman_clean[friedman_clean["significant"]]["metabolite"].nunique())
    n_insignificant = total_points - n_significant

    insig = friedman_clean[~friedman_clean["significant"]]
    sig = friedman_clean[friedman_clean["significant"]]

    fig = go.Figure()
    eps = 1e-12

    def safe_log10_series(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.where(s > 0, np.nan).apply(np.log10)

    def safe_neglog10p(pseries):
        p = pd.to_numeric(pseries, errors="coerce").fillna(1.0)
        p = p.clip(lower=eps)
        return -np.log10(p)

    fig.add_trace(go.Scatter(
        x=safe_log10_series(insig["statistic"]),
        y=safe_neglog10p(insig["p"]),
        mode="markers",
        marker=dict(color="#696880"),
        name="insignificant",
        text=insig["metabolite"],
        hovertemplate="%{text}",
        showlegend=True,
    ))
    if color_by is not None:
        from src.utils import compute_dominant_groups
        dominant_map, all_groups = compute_dominant_groups(
            list(sig["metabolite"]),
            color_by,
            sample_filter_column=st.session_state.get("friedman_attribute"),
            sample_filter_values=st.session_state.get("friedman_groups"),
        )
        colors = px.colors.qualitative.Plotly
        for gi, group in enumerate(all_groups):
            group_sig = sig[sig["metabolite"].map(lambda m, g=group: dominant_map.get(m) == g)]
            if not group_sig.empty:
                fig.add_trace(go.Scatter(
                    x=safe_log10_series(group_sig["statistic"]),
                    y=safe_neglog10p(group_sig["p"]),
                    mode="markers",
                    marker=dict(color=colors[gi % len(colors)]),
                    name=f"{group}",
                    text=group_sig["metabolite"],
                    hovertemplate="%{text}",
                    showlegend=True,
                ))
    else:
        fig.add_trace(go.Scatter(
            x=safe_log10_series(sig["statistic"]),
            y=safe_neglog10p(sig["p"]),
            mode="markers",
            marker=dict(color="#ef553b"),
            name="significant",
            text=sig["metabolite"],
            hovertemplate="%{text}",
            showlegend=True,
        ))

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"Friedman - {st.session_state.friedman_attribute.upper()}",
            "font_color": "#3E3D53",
        },
        xaxis_title="log10(Chi²)",
        yaxis_title="-log10(p)",
        legend=dict(title="Legend"),
        width=600,
        height=600,
    )
    return fig


@st.cache_resource
def get_friedman_metabolite_boxplot(friedman_df, metabolite):
    attribute = st.session_state.friedman_attribute
    p_value = friedman_df.set_index("metabolite")._get_value(metabolite, "p-corrected")

    df = pd.concat([st.session_state.data[[metabolite]], st.session_state.md[[attribute]]], axis=1).copy()
    if "friedman_groups" in st.session_state:
        df = df[df[attribute].isin(st.session_state.friedman_groups)]

    df = df.reset_index().rename(columns={"index": "filename"})
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)

    feature_map = get_feature_name_map()
    metabolite_name = feature_map.get(metabolite, metabolite) if feature_map else metabolite
    df["metabolite_name"] = metabolite_name
    df["intensity"] = df[metabolite]
    df["hovertext"] = df.apply(
        lambda row: (
            f"filename: {row['filename']}<br>"
            f"attribute&group: {attribute}, {row[attribute]}<br>"
            f"metabolite: {row['metabolite_name']}<br>"
            f"intensity: {row['intensity']}"
        ),
        axis=1,
    )

    try:
        p_value_float = float(p_value)
        p_value_str = f"{p_value_float:.2e}"
    except Exception:
        p_value_float = None
        p_value_str = str(p_value)

    is_significant = p_value_float is not None and p_value_float < 0.05
    significance_text = "Significant" if is_significant else "Insignificant"
    title = f"{significance_text} Metabolite: {metabolite_name}<br>Corrected p-value: {p_value_str}"

    fig = px.box(
        df,
        x=attribute,
        y=metabolite,
        template="plotly_white",
        width=800,
        height=600,
        points="all",
        color=attribute,
        hover_data=None,
        custom_data=[df["hovertext"]],
    )
    fig.update_traces(hovertemplate="%{customdata[0]}")
    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        xaxis_title=attribute,
        yaxis_title="intensity",
    )
    return fig
