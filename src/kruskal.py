import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kruskal
from scipy import stats
import scikit_posthocs as sp
from src.utils import get_feature_name_map

def gen_kruskal_data(group_data, _progress_callback=None):
    total = len(group_data[0].columns)
    for idx, col in enumerate(group_data[0].columns):
        try:
            statistic, p = kruskal(*[df[col] for df in group_data])
            if _progress_callback is not None:
                _progress_callback(idx + 1, total, max(0, total - (idx + 1)))
            yield col, p, statistic
        except ValueError:
            continue

def add_p_correction_to_kruskal(df, correction):
    # add Bonferroni corrected p-values for multiple testing correction
    if "p-corrected" not in df.columns:
        df.insert(2, "p-corrected",
                  pg.multicomp(df["p"].astype(float), method=correction)[1])
    # add significance
    if "significant" not in df.columns:
        df.insert(3, "significant", df["p-corrected"] < 0.05)
    # sort by p-value
    df.sort_values("p", inplace=True)
    return df

def kruskal_wallis(df, attribute, correction, elements, _progress_callback=None):
    combined = pd.concat([df, st.session_state.md], axis=1)
    if elements is not None:
        combined = combined[combined[attribute].isin(elements)]
    groups = combined[attribute].unique()
    metabolite_cols = list(st.session_state.data.columns)
    group_data = [combined[combined[attribute] == group].loc[:, metabolite_cols] for group in groups]

    df = pd.DataFrame(
        np.fromiter(
            gen_kruskal_data(group_data, _progress_callback=_progress_callback),
            dtype=[("metabolite", "U100"), ("p", "f"), ("statistic", "f")],
        )
    )
    df = df.dropna()
    df = add_p_correction_to_kruskal(df, correction)
    df = df[df["metabolite"] != attribute]
    return df

@st.cache_resource
def get_kruskal_plot(kruskal, color_by=None):
    # Only count unique, valid metabolite names (not NaN, not attribute name)
    kruskal_clean = kruskal[kruskal["metabolite"].notna()].copy()
    if "kruskal_attribute" in st.session_state:
        kr_attr = st.session_state.kruskal_attribute
        kruskal_clean = kruskal_clean[kruskal_clean["metabolite"] != kr_attr]

    if "significant" in kruskal_clean.columns:
        kruskal_clean["significant"] = kruskal_clean["significant"].fillna(False).astype(bool)
    else:
        kruskal_clean["significant"] = False

    unique_metabolites = kruskal_clean["metabolite"].unique()
    total_points = len(unique_metabolites)
    n_significant = int(kruskal_clean[kruskal_clean["significant"]]["metabolite"].nunique())
    n_insignificant = total_points - n_significant

    st.write(f"Significant: {n_significant}")
    st.write(f"Insignificant: {n_insignificant}")
    st.write(f"Total data points: {total_points}")

    insig = kruskal_clean[~kruskal_clean["significant"]]
    sig = kruskal_clean[kruskal_clean["significant"]]

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
        showlegend=True
    ))
    if color_by is not None:
        from src.utils import compute_dominant_groups
        dominant_map, all_groups = compute_dominant_groups(list(sig["metabolite"]), color_by)
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
                    showlegend=True
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
            showlegend=True
        ))

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"Kruskal Wallis - {st.session_state.kruskal_attribute.upper()}",
            "font_color": "#3E3D53"
        },
        xaxis_title="log10(H)",
        yaxis_title="-log10(p)",
        legend=dict(title="Legend"),
        width=600,
        height=600
    )
    return fig

@st.cache_resource
def get_metabolite_boxplot(kruskal, metabolite):
    attribute = st.session_state.kruskal_attribute
    p_value = kruskal.set_index("metabolite")._get_value(metabolite, "p-corrected")
    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)[[attribute, metabolite]].copy()

    # Add filename column if available
    df = df.reset_index().rename(columns={"index": "filename"})
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)

    # Get metabolite name from feature map if available
    feature_map = get_feature_name_map()
    metabolite_name = feature_map.get(metabolite, metabolite) if feature_map else metabolite
    df["metabolite_name"] = metabolite_name

    df["intensity"] = df[metabolite]
    df["hovertext"] = df.apply(lambda row: f"filename: {row['filename']}<br>attribute&group: {attribute}, {row[attribute]}<br>metabolite: {row['metabolite_name']}<br>intensity: {row['intensity']}", axis=1)

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

def dunn(df, attribute, elements, correction, _progress_callback=None):
 
    """
    Run Dunn's post-hoc test for the two selected groups (elements)
    on the features that were significant in Kruskal–Wallis.

    df: KW results (must have at least 'metabolite' and ideally 'significant')
    attribute: e.g. st.session_state.kruskal_attribute
    elements: the two groups the user selected for Dunn, e.g. ["Control", "Treatment"]
    correction: e.g. "fdr_bh", "bonferroni", or None / "none"
    """

    # run Dunn ONLY on KW-significant features (big speedup)
    if "significant" in df.columns:
        st.session_state.kw_total = len(df)
        df = df[df["significant"] == True]
    else:
        st.session_state.kw_total = len(df)

    # metabolites we actually have in the data matrix
    all_metabolites = df["metabolite"]
    valid_metabolites = [m for m in all_metabolites if m in st.session_state.data.columns]

    # build the full frame = intensities + grouping column
    full_df = pd.concat(
        [st.session_state.data.loc[:, valid_metabolites],
         st.session_state.md[attribute]],
        axis=1,
    )

    gA, gB = elements[0], elements[1]

    results = []
    n_feats = len(valid_metabolites)

    for i, metabolite in enumerate(valid_metabolites):
        # filter to just these 2 groups and this metabolite 
        filtered_df = (
            full_df[full_df[attribute].isin([gA, gB])][[metabolite, attribute]]
            .dropna()
        )

         # if one group has no data, skip
        if filtered_df[attribute].nunique() < 2:
            # store NaNs so table still has row
            results.append(
                {
                    "contrast": f"{gA}-{gB}",
                    "metabolite": metabolite,
                    "rank_sum_diff": np.nan,
                    "p": np.nan,
                }
            )
            continue

        # run Dunn for this single metabolite (two groups)
        # Don't apply per-metabolite p_adjust here; global correction
        # is applied afterward via add_p_value_correction_to_dunns()
        dunn_result = sp.posthoc_dunn(
            a=filtered_df,
            val_col=metabolite,
            group_col=attribute,
            sort=True,
        )
        
        # extract the p-value for this exact contrast
        if gA in dunn_result.index and gB in dunn_result.columns:
            p_val = float(dunn_result.loc[gA, gB])
        elif gB in dunn_result.index and gA in dunn_result.columns:
            p_val = float(dunn_result.loc[gB, gA])
        else:
            p_val = np.nan
        
        # rank-sum diff 
        values = filtered_df[metabolite].to_numpy()
        groups = filtered_df[attribute].to_numpy()
        ranks = stats.rankdata(values)
        maskA = (groups == gA)
        rank_sum_A = ranks[maskA].sum()
        rank_sum_B = ranks[~maskA].sum()
        rank_sum_diff = rank_sum_A - rank_sum_B

        results.append(
            {
                "contrast": f"{gA}-{gB}",
                "metabolite": metabolite,
                "rank_sum_diff": rank_sum_diff,
                "p": p_val,
            }
        )

         # progress callback
        if _progress_callback is not None:
            _progress_callback(i + 1, n_feats, n_feats - (i + 1))

    # make dataframe
    dunn_df = pd.DataFrame(results)

    # global p-correction (only if user asked)
    dunn_df["p"] = pd.to_numeric(dunn_df["p"], errors="coerce")
    dunn_df["rank_sum_diff"] = pd.to_numeric(dunn_df["rank_sum_diff"], errors="coerce")
    
    if correction and correction.lower() != "none":
        dunn_df = add_p_value_correction_to_dunns(dunn_df, correction)
        pcol = "p-corrected"
    else:
        if "p-corrected" not in dunn_df.columns:
            dunn_df["p-corrected"] = dunn_df["p"]
        pcol = "p-corrected"

    dunn_df["significant"] = dunn_df[pcol] < 0.05
    dunn_df = dunn_df.sort_values(pcol)

    st.session_state.dunn_n = len(dunn_df)
    # keep numeric copy
    dunn_original = dunn_df.copy()
    
    # display version
    dunn_display = dunn_df.copy()
    if "p" in dunn_display.columns:
        dunn_display["p"] = dunn_display["p"].apply(
            lambda x: f"{x:.2e}" if pd.notnull(x) else x
            )
    if "p-corrected" in dunn_display.columns:
        dunn_display["p-corrected"] = dunn_display["p-corrected"].apply(
            lambda x: f"{x:.2e}" if pd.notnull(x) else x
            )

    dunn_display._original = dunn_original
    return dunn_display

def gen_pairwise_dunn(group_data, _progress_callback=None):
    """
    Yield results for pairwise dunn test for all metabolites 
    between two options within the attribute.
    group_data: list of 2 DataFrames (one per group, same columns).
    """
    total = len(group_data[0].columns)
    for idx, col in enumerate(group_data[0].columns):
         # Extract values for each group, dropping NaNs
        arrays = [df[col].dropna() for df in group_data]

        # Run Dunn only if both groups have valid data
        if len(arrays[0]) > 0 and len(arrays[1]) > 0:
            try:
                p = float(sp.posthoc_dunn(arrays).iloc[0, 1])
            except Exception:
                p = float("nan")
        else:
            p = float("nan")
        
        if _progress_callback is not None:
            _progress_callback(idx + 1, total, max(0, total - (idx + 1)))
        yield (col, p)


def add_p_value_correction_to_dunns(dunn, correction):
    
    dunn["p"] = pd.to_numeric(dunn["p"], errors="coerce")

    if "p-corrected" not in dunn.columns:
        if correction and correction.lower() != "none":
            # add corrected p-values
            dunn.insert(
                2, 
                "p-corrected", 
                pg.multicomp(dunn["p"], method=correction)[1]
            )
        else:
            dunn.insert(2, "p-corrected", dunn["p"])

        # add significance
        dunn.insert(3, "significant", dunn["p-corrected"] < 0.05)
    
    # sort by p-value
    sort_col = "p-corrected" if "p-corrected" in dunn.columns else "p"
    dunn.sort_values(sort_col, inplace=True)

    return dunn

def _get_dunn_feature_map(df_dunn):
    """Look up feature name mapping for stats_metabolite similar to kruskal."""
    return get_feature_name_map()

@st.cache_resource
def get_dunn_teststat_plot(df, color_by=None):
    feature_map = _get_dunn_feature_map(df)
    fig = go.Figure()

    def make_hovertext(metabolites):
        htext = []
        for m in metabolites:
            met_name = feature_map[m] if feature_map and m in feature_map else str(m)
            htext.append(f"metabolite: {met_name}")
        return htext

    # numeric data
    df_numeric = getattr(df, '_original', df).copy()
    
    sig_col = "significant"
    met_col = "metabolite"
    diff_col = "rank_sum_diff"

    if diff_col not in df_numeric.columns:
        st.error("Rank-sum difference column is not found in Dunn's results. Please rerun the analysis.")
        return fig
    
    if sig_col not in df_numeric.columns:
        st.warning("Significance column not found, showing all points as insignificant.")
        df_numeric[sig_col] = False
    
    if "p-corrected" in df_numeric.columns:
        p_col = "p-corrected"
    elif "p_adj" in df_numeric.columns:
        p_col = "p_adj"
    else:
        p_col = "p"

     # small epsilon to avoid -log10(0)
    eps = 1e-12
    df_numeric["neglog10p"] = -np.log10(df_numeric[p_col].astype(float) + eps)

    # Insignificant points
    ins = df_numeric[df_numeric[sig_col] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=ins[diff_col],
                 y=ins["neglog10p"],
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                hovertext=make_hovertext(ins[met_col]),
                hoverinfo="text",
            )
        )

    # Significant points
    sig = df_numeric[df_numeric[sig_col] == True]
    if not sig.empty:
        if color_by is not None:
            from src.utils import compute_dominant_groups
            dominant_map, all_groups = compute_dominant_groups(list(sig[met_col]), color_by)
            colors = px.colors.qualitative.Plotly
            for gi, group in enumerate(all_groups):
                group_mask = sig[met_col].map(lambda m, g=group: dominant_map.get(m) == g)
                group_sig = sig[group_mask]
                if not group_sig.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=group_sig[diff_col],
                            y=group_sig["neglog10p"],
                            mode="markers",
                            marker=dict(color=colors[gi % len(colors)]),
                            name=f"{group}",
                            hovertext=make_hovertext(group_sig[met_col]),
                            hoverinfo="text",
                        )
                    )
        else:
            fig.add_trace(
                go.Scatter(
                    x=sig[diff_col],
                    y=sig["neglog10p"],
                    mode="markers+text",
                    marker=dict(color="#ef553b"),
                    text=["" for _ in sig[met_col]],
                    textposition="top right",
                    textfont=dict(color="#ef553b", size=12),
                    name="significant",
                    hovertext=make_hovertext(sig[met_col]),
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"DUNN - {st.session_state.kruskal_attribute.upper()}: {st.session_state.dunn_elements[0]} - {st.session_state.dunn_elements[1]} (test-statistic)",
            "font_color": "#3E3D53",
        },
        xaxis_title="Difference of rank sums (A − B)",
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
        width=600,
        height=600,
    )
    fig.update_yaxes(title_standoff=10)
    return fig

@st.cache_resource
def get_dunn_volcano_plot(df):
    """Volcano plot for Dunn's: x = log2 fold change (mean(B)/mean(A)), y = -log10(p or p-corrected)."""
    feature_map = _get_dunn_feature_map(df)
    df_numeric = df.copy()
    eps = 1e-9

    gA, gB = st.session_state.dunn_elements  # e.g. "Control", "Treatment"

    # rebuild means from original data + metadata
    full_df = pd.concat([st.session_state.data, st.session_state.md], axis=1)

    meanA_by_feat = (
        full_df.loc[full_df[st.session_state.kruskal_attribute] == gA, st.session_state.data.columns]
        .mean(axis=0)
    )
    meanB_by_feat = (
        full_df.loc[full_df[st.session_state.kruskal_attribute] == gB, st.session_state.data.columns]
        .mean(axis=0)
    )

    met_col = "stats_metabolite" if "stats_metabolite" in df_numeric.columns else "metabolite"

    # cast to str to match indices
    df_numeric[met_col] = df_numeric[met_col].astype(str)
    meanA_by_feat.index = meanA_by_feat.index.astype(str)
    meanB_by_feat.index = meanB_by_feat.index.astype(str)

    # ❗ map but DO NOT fill with 0 → drop unmapped
    df_numeric["mean(A)"] = df_numeric[met_col].map(meanA_by_feat)
    df_numeric["mean(B)"] = df_numeric[met_col].map(meanB_by_feat)
    df_numeric = df_numeric.dropna(subset=["mean(A)", "mean(B)"])

    # x-axis
    df_numeric["log2FC"] = np.log2((df_numeric["mean(B)"] + eps) / (df_numeric["mean(A)"] + eps))

    # p-axis
    p_col = (
        "p-corrected"
        if "p-corrected" in df_numeric.columns
        else ("p_adj" if "p_adj" in df_numeric.columns else "p")
    )
    df_numeric["neglog10p"] = -np.log10(df_numeric[p_col].astype(float) + eps)

    sig_col = "stats_significant" if "stats_significant" in df_numeric.columns else "significant"

    fig = go.Figure()

    def make_hovertext(metabolites):
        htext = []
        for m in metabolites:
            met_name = feature_map[m] if feature_map and m in feature_map else str(m)
            htext.append(f"metabolite: {met_name}")
        return htext

    # insignificant
    ins = df_numeric[df_numeric[sig_col] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=ins["log2FC"],
                y=ins["neglog10p"],
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                hovertext=make_hovertext(ins[met_col]),
                hoverinfo="text",
            )
        )

    # significant
    sig = df_numeric[df_numeric[sig_col] == True]
    if not sig.empty:
        sig_A = sig[sig["log2FC"] < 0]
        if not sig_A.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_A["log2FC"],
                    y=sig_A["neglog10p"],
                    mode="markers",
                    marker=dict(color="#1f77b4"),
                    name=f"Significant: {gA} > {gB}",
                    hovertext=make_hovertext(sig_A[met_col]),
                    hoverinfo="text",
                )
            )
        sig_B = sig[sig["log2FC"] > 0]
        if not sig_B.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_B["log2FC"],
                    y=sig_B["neglog10p"],
                    mode="markers",
                    marker=dict(color="#ef553b"),
                    name=f"Significant: {gB} > {gA}",
                    hovertext=make_hovertext(sig_B[met_col]),
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"Dunn's post hoc – {st.session_state.kruskal_attribute.upper()}: {gA} vs {gB}",
            "font_color": "#3E3D53",
        },
        xaxis_title="log2(mean B / mean A)",
        yaxis_title="-log10(p)",
        template="plotly_white",
        width=700,
        height=600,
    )
    return fig