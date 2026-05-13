import streamlit as st
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np
import time

def gen_ttest_data(ttest_attribute, target_groups, paired, alternative, correction, p_correction, _progress_callback=None):
    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    ttest = []
    columns = []
    for col in st.session_state.data.columns:
        tmp = df[[ttest_attribute, col]].dropna(subset=[ttest_attribute, col])
        present_groups = set(tmp[ttest_attribute].astype(str).unique())
        if all(str(g) in present_groups for g in target_groups):
            columns.append(col)
    total = len(columns)
    st.session_state.ttest_attempted_metabolites = total
    start_time = None
    for idx, col in enumerate(columns):
        if idx == 0:
            start_time = time.time()
        
        group1 = df[col][df[ttest_attribute] == target_groups[0]]
        group2 = df[col][df[ttest_attribute] == target_groups[1]]
        
        # Calculate means for volcano plot
        mean1 = group1.mean()
        mean2 = group2.mean()
        n1 = len(group1)
        n2 = len(group2)

        # Determine which t-test to use
        # The 'correction' variable is "auto", "True", or "False" (strings)
        correction_param = correction
        if correction == "True":
            correction_param = True
        elif correction == "False":
            correction_param = False
        elif correction == "auto":
            correction_param = True        # Default to Welch

        # Calculate t-test
        result = pg.ttest(group1, group2, paired, alternative, correction=correction_param)

        # Label which test type was used
        if correction_param == True:
            result["ttest_type"] = "Welch"
        elif paired:
            result["ttest_type"] = "Paired Student"
        else:
            result["ttest_type"] = "Student"
        
        result["metabolite"] = col
        result["mean(A)"] = mean1
        result["mean(B)"] = mean2
        ttest.append(result)
        
        # Progress callback with estimated time left
        if _progress_callback is not None:
            elapsed = time.time() - start_time if start_time else 0
            done = idx + 1
            est_left = (elapsed / done) * (total - done) if done > 0 else 0
            _progress_callback(done, total, est_left)

    if not ttest:
        st.session_state.ttest_returned_metabolites = 0
        return pd.DataFrame()

    ttest = pd.concat(ttest).set_index("metabolite")
    ttest = ttest.dropna(subset=['p-val']) # Only drop if p-val is NaN
    st.session_state.ttest_returned_metabolites = len(ttest)

    ttest.insert(8, "p-corrected", pg.multicomp(ttest["p-val"].astype(float), method=p_correction)[1])
    # add significance
    ttest.insert(9, "significance", ttest["p-corrected"] < 0.05)
    ttest.insert(10, "attribute", ttest_attribute)
    ttest.insert(11, "A", target_groups[0])
    ttest.insert(12, "B", target_groups[1])

    # Clean up data types to avoid serialization issues
    ttest = _clean_ttest_dataframe(ttest)

    return ttest.sort_values("p-corrected")

def _clean_ttest_dataframe(df):
    """Clean up t-test dataframe to avoid Arrow serialization issues."""
    df = df.copy()
    
    # Ensure all numeric columns are proper numeric types
    numeric_cols = ["p-val", "p-corrected", "mean(A)", "mean(B)", "dof"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure boolean columns are proper boolean types
    bool_cols = ["significance"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Ensure string columns are proper string types
    str_cols = ["ttest_type", "attribute", "A", "B"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


@st.cache_resource
def plot_ttest(df, color_by=None):

    # Use the correct t-statistic column name from pingouin output
    t_col = None
    for candidate in ["T", "T-val", "t", "tval"]:
        if candidate in df.columns:
            t_col = candidate
            break
    if t_col is None:
        st.error("No t-statistic column found in t-test results. Columns: " + str(df.columns))
        return go.Figure()


    # Add a column for -log(p-corrected) and significance label
    df = df.copy()
    df["-log_p_corrected"] = df["p-corrected"].apply(lambda x: -np.log(x + 1e-300)) # Add epsilon
    if color_by is not None:
        from src.utils import compute_dominant_groups
        sig_mets = list(df[df["significance"]].index)
        dominant_map, all_groups = compute_dominant_groups(
            sig_mets,
            color_by,
            sample_filter_column=st.session_state.get("ttest_attribute"),
            sample_filter_values=st.session_state.get("ttest_options"),
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
        st.warning("No t-test results to display. Please check your data or selection.")
        return go.Figure()

    fig = px.scatter(
        df,
        x=t_col,
        y="-log_p_corrected",
        color="sig_label",
        color_discrete_map=_color_map,
        custom_data=["metabolite_name"],
        template="plotly_white",
        width=600,
        height=600,
    )

    # Custom hovertemplate: metabolite&name: <metabolite name>
    fig.update_traces(hovertemplate="metabolite&name: %{customdata[0]}<extra></extra>")

    xlim = [df[t_col].min(), df[t_col].max()]
    x_padding = abs(xlim[1] - xlim[0]) / 5 if (xlim[1] != xlim[0] and pd.notnull(xlim[0]) and pd.notnull(xlim[1])) else 1
    fig.update_layout(xaxis=dict(range=[xlim[0] - x_padding, xlim[1] + x_padding]))

    # Defensive: Only set title if df has at least one row
    if len(df) > 0:
        title_text = f"t-test - FEATURE SIGNIFICANCE - {str(df.iloc[0, 10]).upper()}: {df.iloc[0, 11]} - {df.iloc[0, 12]}"
    else:
        title_text = "t-test - FEATURE SIGNIFICANCE"
    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": title_text,
            "font_color": "#3E3D53",
        },
        xaxis_title="T-statistic",
        yaxis_title="-Log(p-corrected)",
        showlegend=True,  # Enable legend
        legend_title_text="Significance",
    )
    return fig

def _get_ttest_feature_map():
    """Return a mapping metabolite_id -> feature name. Look for common name columns
    in st.session_state.ft_gnps. If nothing found, return None."""
    ft = st.session_state.get("ft_gnps", pd.DataFrame())
    if ft is None or ft.empty:
        return None
    candidates = ["metabolite_name", "name", "feature_name", "compound_name", "compound"]
    for c in candidates:
        if c in ft.columns:
            return ft[c].to_dict()
    return None

@st.cache_resource(show_spinner="Creating volcano plot...")
def get_ttest_volcano_plot(df):
    """Volcano plot for t-test: x = log2 fold change (mean(B)/mean(A)), y = -log10(p-corrected).
    Adds metabolite/feature hover labels.
    """
    feature_map = _get_ttest_feature_map()
    df = df.copy()
    eps = 1e-9
    
    # Ensure means are numeric, handle potential NaNs
    meanA = pd.to_numeric(df["mean(A)"], errors='coerce').fillna(0) + eps
    meanB = pd.to_numeric(df["mean(B)"], errors='coerce').fillna(0) + eps
    p_values = pd.to_numeric(df["p-corrected"], errors='coerce').fillna(1.0) + eps

    df["log2FC"] = np.log2(meanB / meanA)
    df["neglog10p"] = -np.log10(p_values)

    fig = go.Figure()

    def make_hovertext(metabolites):
        htext = []
        for m in metabolites:
            met_name = feature_map.get(m, str(m)) if feature_map else str(m)
            htext.append(f"metabolite&name: {met_name}")
        return htext

    ins = df[df["significance"] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=ins["log2FC"],
                y=ins["neglog10p"],
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                hovertext=make_hovertext(ins.index),
                hoverinfo="text",
            )
        )

    sig = df[df["significance"] == True]
    if not sig.empty:
        # Group A blue (log2FC < 0)
        sig_A = sig[sig["log2FC"] < 0]
        if not sig_A.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_A["log2FC"],
                    y=sig_A["neglog10p"],
                    mode="markers",
                    marker=dict(color="#1f77b4"), 
                    name=f"Significant: {st.session_state.ttest_options[0]} > {st.session_state.ttest_options[1]}",
                    hovertext=make_hovertext(sig_A.index),
                    hoverinfo="text",
                )
            )
        # Group B red (log2FC > 0)
        sig_B = sig[sig["log2FC"] > 0]
        if not sig_B.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_B["log2FC"],
                    y=sig_B["neglog10p"],
                    mode="markers",
                    marker=dict(color="#ef553b"),
                    name=f"Significant: {st.session_state.ttest_options[1]} > {st.session_state.ttest_options[0]}",
                    hovertext=make_hovertext(sig_B.index),
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"t-test - VOLCANO PLOT - {st.session_state.ttest_attribute.upper()}: {st.session_state.ttest_options[0]} - {st.session_state.ttest_options[1]}",
            "font_color": "#3E3D53",
        },
        xaxis_title="log2(mean B / mean A)",
        yaxis_title="-log10(p-corrected)",
        template="plotly_white",
        legend=dict(
            title="Legend",
            itemsizing='trace'
        ),
        width=700,
        height=600,
    )
    return fig


@st.cache_resource
def ttest_boxplot(df_ttest, metabolite):
    attribute = st.session_state.ttest_attribute
    p_value = df_ttest.loc[metabolite, "p-corrected"]

    df = pd.concat([st.session_state.data[[metabolite]], st.session_state.md[[attribute]]], axis=1).copy()

    # Filter for the two selected groups and ensure only those two are present, in order
    options = st.session_state.ttest_options
    df = df[df[attribute].isin(options)].copy()
    df[attribute] = pd.Categorical(df[attribute], categories=options, ordered=True)
    df = df.reset_index().rename(columns={"index": "filename"})
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)

    feature_map = _get_ttest_feature_map()
    metabolite_name = feature_map.get(metabolite, metabolite) if feature_map else metabolite
    df["metabolite_name"] = metabolite_name

    df["intensity"] = df[metabolite]
    df["hovertext"] = df.apply(lambda row: f"filename: {row['filename']}<br>attribute&group: {attribute}, {row[attribute]}<br>metabolite&name: {row['metabolite_name']}<br>intensity: {row['intensity']}", axis=1)

    try:
        p_value_float = float(p_value)
        p_value_str = f"{p_value_float:.2e}"
    except Exception:
        p_value_float = None
        p_value_str = str(p_value)

    # Determine significance
    is_significant = p_value_float is not None and p_value_float < 0.05
    significance_text = "Significant" if is_significant else "Insignificant"
    title = f"{significance_text} Metabolite: {metabolite_name}<br>Corrected p-value: {p_value_str}"
    
    # Create the plot from the filtered dataframe
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
