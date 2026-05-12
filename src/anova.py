import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import time
from src.utils import get_feature_name_map

if 'name_column' not in st.session_state:
    st.session_state['name_column'] = None

def gen_anova_data(df, columns, groups_col, _progress_callback=None):
    """
    Robustly run pg.anova for each column in `columns` and yield
    (metabolite, p-value, F-value). This function tolerates variations
    in pingouin output column/row names.
    """
    total = len(columns)
    start_time = time.time()
    for idx, col in enumerate(columns):
        iter_start = time.time()
        try:
            result = pg.anova(data=df, dv=col, between=groups_col, detailed=True)
        except Exception as e:
            st.warning(f"ANOVA failed for {col}: {e}")
            continue

        row = None
        if "Source" in result.columns:
            matches = result[result["Source"].astype(str) == str(groups_col)]
            if not matches.empty:
                row = matches.iloc[0]
            else:
                invalid = {"residual", "residuals", "within", "error", "intercept"}
                candidates = result[~result["Source"].astype(str).str.lower().isin(invalid)]
                if not candidates.empty:
                    row = candidates.iloc[0]
                else:
                    row = result.iloc[0]
        else:
            row = result.iloc[0]

        p = None
        p_candidates = ["p"]
        for pc in p_candidates:
            if pc in result.columns:
                try:
                    p = float(row[pc])
                except Exception:
                    p = row[pc]
                break
        if p is None:
            for c in result.columns:
                if "p" in c.lower():
                    try:
                        p = float(row[c])
                    except Exception:
                        p = row[c]
                    break

        f = None
        f_candidates = ["F"]
        for fc in f_candidates:
            if fc in result.columns:
                try:
                    f = float(row[fc])
                except Exception:
                    f = row[fc]
                break
        if f is None:
            for c in result.columns:
                if c.lower().startswith("f"):
                    try:
                        f = float(row[c])
                    except Exception:
                        f = row[c]
                    break

        if p is None or f is None:
            continue

        # Progress callback
        if _progress_callback is not None:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            est_total = avg_time * total
            est_left = max(0, est_total - elapsed)
            _progress_callback(idx + 1, total, est_left)

        yield col, p, f

def add_p_correction_to_anova(df, correction):
    # add Bonferroni corrected p-values for multiple testing correction
    if "p-corrected" not in df.columns:
        df.insert(2, "p-corrected",
                  pg.multicomp(df["p"].astype(float), method=correction)[1])
    # add significance
    if "significant" not in df.columns:
        df.insert(3, "significant", df["p-corrected"] < 0.05)
    
    df.sort_values("p", inplace=True)
    return df

def anova(df, attribute, correction, elements, _progress_callback=None):
    """
    Run ANOVA on metabolite columns in `df` using the metadata attribute `attribute`.
    If `elements` is provided (list of category values), only samples whose metadata
    attribute is in `elements` are included in the test.
    """
    combined = pd.concat([df, st.session_state.md], axis=1)

    if elements is not None:
        combined = combined[combined[attribute].isin(elements)]

    target_groups = list(elements) if elements is not None else list(combined[attribute].dropna().unique())
    if not target_groups:
        st.session_state.anova_attempted_metabolites = 0
        st.session_state.anova_returned_metabolites = 0
        return pd.DataFrame(columns=["p", "F", "p-corrected", "significant"])

    valid_metabolites = []
    for col in df.columns:
        tmp = combined[[attribute, col]].dropna(subset=[attribute, col])
        if tmp.empty:
            continue
        present_groups = set(tmp[attribute].astype(str).unique())
        if all(str(g) in present_groups for g in target_groups):
            valid_metabolites.append(col)

    st.session_state.anova_attempted_metabolites = len(valid_metabolites)
    results = list(gen_anova_data(combined, valid_metabolites, attribute, _progress_callback=_progress_callback))
    df_res = pd.DataFrame(results, columns=["metabolite", "p", "F"])
    df_res = df_res.dropna()
    st.session_state.anova_returned_metabolites = len(df_res)
    df_res = add_p_correction_to_anova(df_res, correction)
    return df_res.set_index("metabolite")

@st.cache_resource(show_spinner="Creating ANOVA plot...")
def get_anova_plot(anova, color_by=None):
    """ANOVA scatter: x=log(F), y=-log(p). Add hover text with feature name if available."""
    feature_map = get_feature_name_map()
    

    def create_hovertexts(indexes):
        hover = []
        # Generate hovertexts for each metabolite index
        for m in indexes:
            metabolite_name = feature_map[m] if feature_map and m in feature_map else str(m)
            hover.append(f"metabolite&name: {metabolite_name}")
        
        return hover

    fig = go.Figure()

    total_points = len(anova)
    n_significant = int(anova["significant"].sum())
    n_insignificant = total_points - n_significant

    ins = anova[anova["significant"] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=np.log(ins["F"]),
                y=-np.log(ins["p"]),
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                hovertext=create_hovertexts(ins.index),
                hoverinfo="text",
            )
        )

    sig = anova[anova["significant"] == True]
    if not sig.empty:
        if color_by is not None:
            from src.utils import compute_dominant_groups
            dominant_map, all_groups = compute_dominant_groups(
                list(sig.index),
                color_by,
                sample_filter_column=st.session_state.get("anova_attribute"),
                sample_filter_values=st.session_state.get("anova_groups"),
            )
            colors = px.colors.qualitative.Plotly
            for gi, group in enumerate(all_groups):
                group_sig = sig[sig.index.map(lambda m, g=group: dominant_map.get(m) == g)]
                if not group_sig.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=np.log(group_sig["F"]),
                            y=-np.log(group_sig["p"]),
                            mode="markers",
                            marker=dict(color=colors[gi % len(colors)]),
                            name=f"{group}",
                            hovertext=create_hovertexts(group_sig.index),
                            hoverinfo="text",
                        )
                    )
        else:
            fig.add_trace(
                go.Scatter(
                    x=np.log(sig["F"]),
                    y=-np.log(sig["p"]),
                    mode="markers",
                    marker=dict(color="#ef553b"),
                    name="significant",
                    hovertext=create_hovertexts(sig.index),
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"ANOVA - {st.session_state.anova_attribute.upper()}",
            "font_color": "#3E3D53"
        },
        xaxis_title="log(F)",
        yaxis_title="-log(p)",
        showlegend=True,
        legend=dict(
            itemsizing='trace',
            font=dict(size=12),
            orientation="v"
        ),
        template="plotly_white",
        width=600,
        height=600,
    )
    fig.update_yaxes(title_standoff=10)
    return fig

@st.cache_resource(show_spinner="Creating metabolite boxplot...")
def get_metabolite_boxplot(anova, metabolite):
    """Build a boxplot for *any* metabolite (not only significant ones).
    Adds per-sample hover that includes filename or sample id.
    NOTE: Now restricts plotting to the groups selected for ANOVA (st.session_state.anova_groups)
    if that session variable exists.
    """
    attribute = st.session_state.anova_attribute
    p_value = anova.loc[metabolite, "p-corrected"]

    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)[[attribute, metabolite]].copy()

    if "anova_groups" in st.session_state and st.session_state.anova_groups:
        df = df[df[attribute].isin(st.session_state.anova_groups)]

    df = df.reset_index().rename(columns={"index": "filename"})
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)

    feature_map = get_feature_name_map()
    metabolite_name = feature_map.get(metabolite, metabolite) if feature_map else metabolite
    df["metabolite_name"] = metabolite_name

    df["intensity"] = df[metabolite]
    df["hovertext"] = df.apply(lambda row: f"filename: {row['filename']}<br>attribute&group: {attribute}, {row[attribute]}<br>metabolite&name: {row['metabolite_name']}<br>intensity: {row['intensity']}", axis=1)

    try:
        p_value_str = f"{float(p_value):.2e}"
    except Exception:
        p_value_str = str(p_value)

    title = f"Corrected p-value: {p_value_str}"
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

def gen_pairwise_tukey(df, _metabolites, attribute, _progress_callback=None):
    """Return a list of results for pairwise Tukey test for all metabolites between two options 
    within the attribute."""
    
    import time
    results = []
    total = len(_metabolites)
    start_time = time.time()

    for idx, metabolite in enumerate(_metabolites):
        if _progress_callback is not None:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            est_total = avg_time * total
            est_left = max(0, est_total - elapsed)
            _progress_callback(idx + 1, total, est_left)

        try:
            tukey = pg.pairwise_tukey(df, dv=metabolite, between=attribute)
        except Exception as e:
            continue

        if tukey.empty:
            continue
        results.append(
            (
            metabolite,
            tukey.loc[0, "diff"],
            tukey.loc[0, "p_tukey"],
            attribute,
            tukey.loc[0, "A"],
            tukey.loc[0, "B"],
            tukey.loc[0, "mean_A"],
            tukey.loc[0, "mean_B"],
            )
        )
    return results

def tukey(df, attribute, elements, correction, _progress_callback=None):
    """
    Run Tukey's test for all significant metabolites, with progress callback.
    """
    # run Tukey ONLY on anova significant features
    st.session_state.anova_total = len(df)

    if "significant" in df.columns:
        df = df[df["significant"] == True]
    
    # metabolites we actually have in the data matrix
    if "metabolite" in df.columns:
        anova_mets = df["metabolite"].astype(str).tolist()
    else:
        # fallback: some ANOVA functions return feature names as index
        anova_mets = df.index.astype(str).tolist()

    valid_metabolites = [m for m in anova_mets if m in st.session_state.data.columns]
    if elements is not None:
        md_attr = st.session_state.md.loc[:, attribute]
        filtered_valid = []
        for m in valid_metabolites:
            tmp = pd.concat([st.session_state.data.loc[:, m], md_attr], axis=1)
            tmp = tmp[tmp[attribute].isin(elements)]
            tmp = tmp.dropna(subset=[m, attribute])
            present_groups = set(tmp[attribute].astype(str).unique())
            if all(str(g) in present_groups for g in elements):
                filtered_valid.append(m)
        valid_metabolites = filtered_valid
    st.session_state.tukey_attempted_metabolites = len(valid_metabolites)

    if not valid_metabolites:
        st.session_state.tukey_returned_metabolites = 0
        # nothing to run Tukey on
        return pd.DataFrame()

    data = pd.concat([st.session_state.data.loc[:, valid_metabolites],
            st.session_state.md.loc[:, attribute]], axis=1)
    data = data[data[attribute].isin(elements)]
    
    tukey = pd.DataFrame(
        np.array(
            gen_pairwise_tukey(
                data, valid_metabolites, attribute, _progress_callback=_progress_callback
                ),
                dtype=[("stats_metabolite", "U100"), (f"diff", "f"), (f"stats_p", "f"), ("attribute", "U100"), ("A", "U100"), ("B", "U100"), ("mean_A", "f"), ("mean_B", "f"),],
        )
    )

    tukey = tukey.dropna()
    st.session_state.tukey_returned_metabolites = len(tukey)
    tukey = add_p_value_correction_to_tukeys(tukey, correction)

    tukey = tukey.rename(columns={ "stats_metabolite": "metabolite", "stats_p": "p", "stats_significant": "significant"})
    tukey_display = tukey.copy()
    st.session_state.tukey_n = len(tukey_display)

    if "p" in tukey_display.columns:
        tukey_display["p"] = tukey_display["p"].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else x)
    
    if "diff" in tukey_display.columns:
        cols = [c for c in tukey_display.columns if c != "diff"] + ["diff"]
        tukey_display = tukey_display[cols]
    
    tukey_display._original = tukey  
    return tukey_display

def add_p_value_correction_to_tukeys(tukey, correction):
    # ensure numeric
    tukey["stats_p"] = pd.to_numeric(tukey["stats_p"], errors="coerce")

    if "p-corrected" not in tukey.columns:
        if correction and correction.lower() != "none":
            tukey.insert(3, "p-corrected", pg.multicomp(tukey["stats_p"].astype(float), method=correction)[1])
        else:
            tukey.insert(3, "p-corrected", tukey["stats_p"])

        tukey.insert(4, "stats_significant", tukey["p-corrected"] < 0.05)
        tukey.sort_values("stats_p", inplace=True)
    return tukey

def _get_tukey_feature_map(df_tukey):
    """Look up feature name mapping for stats_metabolite similar to anova."""
    return get_feature_name_map()

@st.cache_resource(show_spinner="Creating Tukey test statistic plots...")
def get_tukey_teststat_plot(df, color_by=None):
    """Plot the test-statistic/diff (existing behaviour) but add hover text."""
    feature_map = _get_tukey_feature_map(df)
    fig = go.Figure()

    sample_ids = list(st.session_state.data.index) if hasattr(st.session_state.data, 'index') else None
    def make_hovertext(metabolites):
        htext = []
        brk = "<br>"
        for i, m in enumerate(metabolites):
            met_name = feature_map[m] if feature_map and m in feature_map else str(m)
            filename = sample_ids[i % len(sample_ids)] if sample_ids else "N/A"
            htext.append(f"filename: {filename}<br>metabolite: {met_name}")
        return htext

    p_numeric = getattr(df, '_original', df)["p"] if hasattr(df, '_original') else df["p"]
    ins = df[df["significant"] == False]
    if not ins.empty:
        ins_numeric = p_numeric[ins.index]
        fig.add_trace(
            go.Scatter(
                x=ins["diff"],
                y=-np.log(ins_numeric.astype(float)),
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                hovertext=make_hovertext(ins["metabolite"]),
                hoverinfo="text",
            )
        )

    sig = df[df["significant"] == True]
    if not sig.empty:
        sig_numeric = p_numeric[sig.index]
        if color_by is not None:
            from src.utils import compute_dominant_groups
            dominant_map, all_groups = compute_dominant_groups(
                list(sig["metabolite"]),
                color_by,
                sample_filter_column=st.session_state.get("anova_attribute"),
                sample_filter_values=st.session_state.get("tukey_elements"),
            )
            colors = px.colors.qualitative.Plotly
            for gi, group in enumerate(all_groups):
                group_mask = sig["metabolite"].map(lambda m, g=group: dominant_map.get(m) == g)
                group_sig = sig[group_mask]
                if not group_sig.empty:
                    group_sig_p = sig_numeric[group_sig.index]
                    fig.add_trace(
                        go.Scatter(
                            x=group_sig["diff"],
                            y=-np.log(group_sig_p.astype(float)),
                            mode="markers",
                            marker=dict(color=colors[gi % len(colors)]),
                            name=f"{group}",
                            hovertext=make_hovertext(group_sig["metabolite"]),
                            hoverinfo="text",
                        )
                    )
        else:
            fig.add_trace(
                go.Scatter(
                    x=sig["diff"],
                    y=-np.log(sig_numeric.astype(float)),
                    mode="markers+text",
                    marker=dict(color="#ef553b"),
                    text=["" for _ in sig["metabolite"]],
                    textposition="top right",
                    textfont=dict(color="#ef553b", size=12),
                    name="significant",
                    hovertext=make_hovertext(sig["metabolite"]),
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"TUKEY - {st.session_state.anova_attribute.upper()}: {st.session_state.tukey_elements[0]} - {st.session_state.tukey_elements[1]} (test-statistic)",
            "font_color": "#3E3D53",
        },
        xaxis_title="diff (mean B - mean A)",
        yaxis_title="-log(p)",
        template="plotly_white",
    )
    return fig

@st.cache_resource(show_spinner="Creating Tukey volcano plot...")
def get_tukey_volcano_plot(df):
    """Volcano plot for Tukey: x = log2 fold change (mean(B)/mean(A)), y = -log10(p-value).
    Adds metabolite/feature hover labels.
    """
    feature_map = _get_tukey_feature_map(df)

    # compute log2 fold change (B relative to A). avoiding zeros by using a small epsilon
    eps = 1e-9
    meanA = df["mean_A"].astype(float) + eps
    meanB = df["mean_B"].astype(float) + eps
    df = df.copy()
    df["log2FC"] = np.log2(meanB/meanA)
    df["neglog10p"] = -np.log10(df["p"].astype(float) + eps)

    fig = go.Figure()

    sample_ids = list(st.session_state.data.index) if hasattr(st.session_state.data, 'index') else None
    def make_hovertext(metabolites):
        htext = []
        for i, m in enumerate(metabolites):
            met_name = feature_map[m] if feature_map and m in feature_map else str(m)
            filename = sample_ids[i % len(sample_ids)] if sample_ids else "N/A"
            htext.append(f"filename: {filename}<br>metabolite: {met_name}")
        return htext

    ins = df[df["significant"] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=ins["log2FC"],
                y=ins["neglog10p"],
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                hovertext=make_hovertext(ins["metabolite"]),
                hoverinfo="text",
            )
        )

    sig = df[df["significant"] == True]
    if not sig.empty:
        # Group A blue
        sig_A = sig[sig["log2FC"] < 0]
        if not sig_A.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_A["log2FC"],
                    y=sig_A["neglog10p"],
                    mode="markers+text",
                    marker=dict(color="#1f77b4"), 
                    text=["" for _ in sig_A["metabolite"]],
                    textposition="top right",
                    textfont=dict(color="#1f77b4", size=12),
                    name=f"Significant: {st.session_state.tukey_elements[0]} > {st.session_state.tukey_elements[1]}",
                    hovertext=make_hovertext(sig_A["metabolite"]),
                    hoverinfo="text",
                )
            )
        # Group B red
        sig_B = sig[sig["log2FC"] > 0]
        if not sig_B.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_B["log2FC"],
                    y=sig_B["neglog10p"],
                    mode="markers+text",
                    marker=dict(color="#ef553b"),  #red
                    text=["" for _ in sig_B["metabolite"]],
                    textposition="top right",
                    textfont=dict(color="#ef553b", size=12),
                    name=f"Significant: {st.session_state.tukey_elements[1]} > {st.session_state.tukey_elements[0]}",
                    hovertext=make_hovertext(sig_B["metabolite"]),
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"TUKEY - {st.session_state.anova_attribute.upper()}: {st.session_state.tukey_elements[0]} - {st.session_state.tukey_elements[1]} (volcano)",
            "font_color": "#3E3D53",
        },
        xaxis_title="log2(mean B/mean A)",
        yaxis_title="-log10(p)",
        template="plotly_white",
        legend=dict(
            title="Legend",
            itemsizing='trace',
            font=dict(size=12),
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        width=700,
        height=600,
    )
    return fig
