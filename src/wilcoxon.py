import pandas as pd
import numpy as np
import streamlit as st
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import time


def gen_wilcoxon_data(wilcoxon_attribute, target_groups, alternative, p_correction, _progress_callback=None):
    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    wilcoxon_results = []
    columns = list(st.session_state.data.columns)
    total = len(columns)
    start_time = None

    for idx, col in enumerate(columns):
        if idx == 0:
            start_time = time.time()

        group1 = df[col][df[wilcoxon_attribute] == target_groups[0]].reset_index(drop=True)
        group2 = df[col][df[wilcoxon_attribute] == target_groups[1]].reset_index(drop=True)

        # Paired test requires equal-length groups; truncate to the shorter one
        min_len = min(len(group1), len(group2))
        if min_len < 2:
            continue
        group1 = group1[:min_len]
        group2 = group2[:min_len]

        median1 = group1.median()
        median2 = group2.median()

        try:
            result = pg.wilcoxon(group1, group2, alternative=alternative)
        except Exception:
            continue

        result["metabolite"] = col
        result["median(A)"] = median1
        result["median(B)"] = median2
        wilcoxon_results.append(result)

        if _progress_callback is not None:
            elapsed = time.time() - start_time if start_time else 0
            done = idx + 1
            est_left = (elapsed / done) * (total - done) if done > 0 else 0
            _progress_callback(done, total, est_left)

    if not wilcoxon_results:
        return pd.DataFrame()

    wilcoxon_df = pd.concat(wilcoxon_results).set_index("metabolite")
    wilcoxon_df = wilcoxon_df.dropna(subset=["p-val"])

    wilcoxon_df.insert(4, "p-corrected", pg.multicomp(wilcoxon_df["p-val"].astype(float), method=p_correction)[1])
    wilcoxon_df.insert(5, "significance", wilcoxon_df["p-corrected"] < 0.05)
    wilcoxon_df.insert(6, "attribute", wilcoxon_attribute)
    wilcoxon_df.insert(7, "A", target_groups[0])
    wilcoxon_df.insert(8, "B", target_groups[1])

    wilcoxon_df = _clean_wilcoxon_dataframe(wilcoxon_df)
    return wilcoxon_df.sort_values("p-corrected")


def _clean_wilcoxon_dataframe(df):
    df = df.copy()
    numeric_cols = ["W-val", "p-val", "p-corrected", "RBC", "CLES", "median(A)", "median(B)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bool_cols = ["significance"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    str_cols = ["attribute", "A", "B"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_resource
def plot_wilcoxon(df, color_by=None):
    df = df.copy()
    df["-log_p_corrected"] = df["p-corrected"].apply(lambda x: -np.log(x + 1e-300))
    if color_by is not None:
        from src.utils import compute_dominant_groups
        sig_mets = list(df[df["significance"]].index)
        dominant_map, all_groups = compute_dominant_groups(sig_mets, color_by)
        df["sig_label"] = df.apply(
            lambda row: dominant_map.get(row.name, "insignificant") if row["significance"] else "insignificant",
            axis=1
        )
        colors = px.colors.qualitative.Plotly
        _color_map = {"insignificant": "#696880"}
        for i, g in enumerate(all_groups):
            _color_map[g] = colors[i % len(colors)]
    else:
        df["sig_label"] = df["significance"].apply(lambda x: "significant" if x else "insignificant")
        _color_map = {"significant": "#ef553b", "insignificant": "#696880"}
    df["metabolite_name"] = df.index

    if df.empty:
        st.warning("No Wilcoxon results to display. Please check your data or selection.")
        return go.Figure()

    fig = px.scatter(
        df,
        x="W-val",
        y="-log_p_corrected",
        color="sig_label",
        color_discrete_map=_color_map,
        custom_data=["metabolite_name"],
        template="plotly_white",
        width=600,
        height=600,
    )
    fig.update_traces(hovertemplate="metabolite&name: %{customdata[0]}<extra></extra>")

    xlim = [df["W-val"].min(), df["W-val"].max()]
    x_padding = (
        abs(xlim[1] - xlim[0]) / 5
        if (xlim[1] != xlim[0] and pd.notnull(xlim[0]) and pd.notnull(xlim[1]))
        else 1
    )
    fig.update_layout(xaxis=dict(range=[xlim[0] - x_padding, xlim[1] + x_padding]))

    if len(df) > 0:
        title_text = (
            f"Wilcoxon Signed-Rank - FEATURE SIGNIFICANCE - "
            f"{str(df.iloc[0, 6]).upper()}: {df.iloc[0, 7]} - {df.iloc[0, 8]}"
        )
    else:
        title_text = "Wilcoxon Signed-Rank - FEATURE SIGNIFICANCE"

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title_text, "font_color": "#3E3D53"},
        xaxis_title="W-statistic",
        yaxis_title="-Log(p-corrected)",
        showlegend=True,
        legend_title_text="Significance",
    )
    return fig


@st.cache_resource
def wilcoxon_boxplot(df_wilcoxon, metabolite):
    attribute = st.session_state.wilcoxon_attribute
    p_value = df_wilcoxon.loc[metabolite, "p-corrected"]

    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)[[attribute, metabolite]].copy()
    options = st.session_state.wilcoxon_options
    df = df[df[attribute].isin(options)].copy()
    df[attribute] = pd.Categorical(df[attribute], categories=options, ordered=True)
    df = df.reset_index().rename(columns={"index": "filename"})
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)

    df["metabolite_name"] = metabolite
    df["intensity"] = df[metabolite]
    df["hovertext"] = df.apply(
        lambda row: (
            f"filename: {row['filename']}<br>"
            f"attribute&group: {attribute}, {row[attribute]}<br>"
            f"metabolite&name: {row['metabolite_name']}<br>"
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
    title = f"{significance_text} Metabolite: {metabolite}<br>Corrected p-value: {p_value_str}"

    fig = px.box(
        df,
        x=attribute,
        y=metabolite,
        color=attribute,
        template="plotly_white",
        points="all",
        hover_data=None,
        custom_data=[df["hovertext"]],
    )
    fig.update_traces(hovertemplate="%{customdata[0]}")
    fig.update_layout(
        boxmode="group",
        xaxis_title=attribute,
        yaxis_title="intensity",
        template="plotly_white",
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        autosize=True,
    )
    fig.update_yaxes(title_standoff=10)
    return fig
