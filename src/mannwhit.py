import pandas as pd
import numpy as np
import streamlit as st
import pingouin as pg
import time

def gen_mwu_data(mwu_attribute, target_groups, alternative, p_correction, _progress_callback=None):
    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    mwu_results = []
    columns = []
    for col in st.session_state.data.columns:
        tmp = df[[mwu_attribute, col]].dropna(subset=[mwu_attribute, col])
        present_groups = set(tmp[mwu_attribute].astype(str).unique())
        if all(str(g) in present_groups for g in target_groups):
            columns.append(col)
    total = len(columns)
    st.session_state.mwu_attempted_metabolites = total
    start_time = None
    for idx, col in enumerate(columns):
        if idx == 0:
            start_time = time.time()
        group1 = df[col][df[mwu_attribute] == target_groups[0]]
        group2 = df[col][df[mwu_attribute] == target_groups[1]]
        # Calculate means for boxplot
        mean1 = group1.mean()
        mean2 = group2.mean()
        n1 = len(group1)
        n2 = len(group2)
        # Mann-Whitney U test
        result = pg.mwu(group1, group2, alternative=alternative)
        result["metabolite"] = col
        result["mean(A)"] = mean1
        result["mean(B)"] = mean2
        mwu_results.append(result)
        # Progress callback
        if _progress_callback is not None:
            elapsed = time.time() - start_time if start_time else 0
            done = idx + 1
            est_left = (elapsed / done) * (total - done) if done > 0 else 0
            _progress_callback(done, total, est_left)

    if not mwu_results:
        st.session_state.mwu_returned_metabolites = 0
        return pd.DataFrame()

    mwu = pd.concat(mwu_results).set_index("metabolite")
    mwu = mwu.dropna(subset=["p-val"])
    st.session_state.mwu_returned_metabolites = len(mwu)
    mwu.insert(4, "p-corrected", pg.multicomp(mwu["p-val"].astype(float), method=p_correction)[1])
    mwu.insert(5, "significance", mwu["p-corrected"] < 0.05)
    mwu.insert(6, "attribute", mwu_attribute)
    mwu.insert(7, "A", target_groups[0])
    mwu.insert(8, "B", target_groups[1])
    mwu = _clean_mwu_dataframe(mwu)
    return mwu.sort_values("p-corrected")

def _clean_mwu_dataframe(df):
    df = df.copy()
    numeric_cols = ["U-val", "p-val", "p-corrected", "mean(A)", "mean(B)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
def plot_mwu(df, color_by=None):
    df = df.copy()
    df["-log_p_corrected"] = df["p-corrected"].apply(lambda x: -np.log(x + 1e-300))
    import plotly.express as px
    import plotly.graph_objects as go
    if color_by is not None:
        from src.utils import compute_dominant_groups
        sig_mets = list(df[df["significance"]].index)
        dominant_map, all_groups = compute_dominant_groups(
            sig_mets,
            color_by,
            sample_filter_column=st.session_state.get("mwu_attribute"),
            sample_filter_values=st.session_state.get("mwu_options"),
        )
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
        st.warning("No MWU results to display. Please check your data or selection.")
        return go.Figure()
    fig = px.scatter(
        df,
        x="U-val",
        y="-log_p_corrected",
        color="sig_label",
        color_discrete_map=_color_map,
        custom_data=["metabolite_name"],
        template="plotly_white",
        width=600,
        height=600,
    )
    fig.update_traces(hovertemplate="metabolite&name: %{customdata[0]}<extra></extra>")
    xlim = [df["U-val"].min(), df["U-val"].max()]
    x_padding = abs(xlim[1] - xlim[0]) / 5 if (xlim[1] != xlim[0] and pd.notnull(xlim[0]) and pd.notnull(xlim[1])) else 1
    fig.update_layout(xaxis=dict(range=[xlim[0] - x_padding, xlim[1] + x_padding]))
    if len(df) > 0:
        title_text = f"MWU - FEATURE SIGNIFICANCE - {str(df.iloc[0, 6]).upper()}: {df.iloc[0, 7]} - {df.iloc[0, 8]}"
    else:
        title_text = "MWU - FEATURE SIGNIFICANCE"
    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": title_text,
            "font_color": "#3E3D53",
        },
        xaxis_title="U-statistic",
        yaxis_title="-Log(p-corrected)",
        showlegend=True,
        legend_title_text="Significance",
    )
    return fig

@st.cache_resource
def mwu_boxplot(df_mwu, metabolite):
    attribute = st.session_state.mwu_attribute
    p_value = df_mwu.loc[metabolite, "p-corrected"]
    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)[[attribute, metabolite]].copy()
    options = st.session_state.mwu_options
    df = df[df[attribute].isin(options)].copy()
    df[attribute] = pd.Categorical(df[attribute], categories=options, ordered=True)
    df = df.reset_index().rename(columns={"index": "filename"})
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)
    df["metabolite_name"] = metabolite
    df["intensity"] = df[metabolite]
    df["hovertext"] = df.apply(lambda row: f"filename: {row['filename']}<br>attribute&group: {attribute}, {row[attribute]}<br>metabolite&name: {row['metabolite_name']}<br>intensity: {row['intensity']}", axis=1)
    import plotly.express as px
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
        boxmode='group',
        xaxis_title=attribute,
        yaxis_title="intensity",
        template="plotly_white",
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": title,
            "font_color": "#3E3D53",
        },
        autosize=True,
    )
    fig.update_yaxes(title_standoff=10)
    return fig
