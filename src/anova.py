import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go


def gen_anova_data(df, columns, groups_col):
    """
    Robustly run pg.anova for each column in `columns` and yield
    (metabolite, p-value, F-value). This function tolerates variations
    in pingouin output column/row names.
    """
    for col in columns:
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

        # If we couldn't find either p or F, skip this metabolite
        if p is None or f is None:
            #st.warning(f"ANOVA returned unexpected table for {col}; skipping (no p or F found).")
            continue

        yield col, p, f


def add_p_correction_to_anova(df, correction):
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


@st.cache_data
def anova(df, attribute, correction, elements):
    """
    Run ANOVA on metabolite columns in `df` using the metadata attribute `attribute`.
    If `elements` is provided (list of category values), only samples whose metadata
    attribute is in `elements` are included in the test.
    """
    # Combine metabolite data (df) with metadata (st.session_state.md)
    combined = pd.concat([df, st.session_state.md], axis=1)

    # If elements specified, filter samples to only those groups
    if elements is not None:
        combined = combined[combined[attribute].isin(elements)]

    # run ANOVA for every metabolite column that exists in df
    df_res = pd.DataFrame(
        np.fromiter(
            gen_anova_data(
                combined,
                df.columns,
                attribute,
            ),
            dtype=[("metabolite", "U100"), ("p", "f"), ("F", "f")],
        )
    )
    df_res = df_res.dropna()
    df_res = add_p_correction_to_anova(df_res, correction)
    return df_res.set_index("metabolite")


def _get_feature_name_map():
    """Return a mapping metabolite_id -> feature name. Look for common name columns
    in st.session_state.ft_gnps. If nothing found, return None."""
    ft = st.session_state.get("ft_gnps", pd.DataFrame())
    if ft is None or ft.empty:
        return None
    # prefer columns named in various common ways
    candidates = ["metabolite_name", "name", "feature_name", "compound_name", "compound"]
    for c in candidates:
        if c in ft.columns:
            # ensure index aligns to metabolite ids used in anova/tukey (assumes ft index contains feature ids)
            # If ft has a column that stores feature id, we still assume ft.index matches the metabolite keys in df_anova
            return ft[c].to_dict()
    return None


@st.cache_resource
def get_anova_plot(anova):
    """ANOVA scatter: x=log(F), y=-log(p). Add hover text with feature name if available."""
    feature_map = _get_feature_name_map()
    # prepare points data
    def create_hovertexts(indexes):
        hover = []
        for m in indexes:
            if feature_map and m in feature_map:
                hover.append(f"{m} — {feature_map[m]}")
            else:
                hover.append(str(m))
        return hover

    fig = go.Figure()

    # insignificant features
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

    # significant features
    sig = anova[anova["significant"] == True]
    if not sig.empty:
        # show labels for top few; hovertext for all
        top_text_idx = sig.index[:6]
        top_texts = [t if not feature_map or t not in feature_map else f"{t} — {feature_map[t]}" for t in top_text_idx]

        fig.add_trace(
            go.Scatter(
                x=np.log(sig["F"]),
                y=-np.log(sig["p"]),
                mode="markers+text",
                marker=dict(color="#ef553b"),
                text=[(t if i < len(top_texts) else "") for i, t in enumerate(sig.index)],
                textposition="top left",
                textfont=dict(color="#ef553b", size=12),
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
            itemsizing='trace',  # valid values: 'trace' or 'constant'
            font=dict(size=12),
            orientation="v"
        ),
        template="plotly_white",
        width=600,
        height=600,
    )
    fig.update_yaxes(title_standoff=10)
    return fig


@st.cache_resource
def get_metabolite_boxplot(anova, metabolite):
    """Build a boxplot for *any* metabolite (not only significant ones).
    Adds per-sample hover that includes filename or sample id.
    NOTE: Now restricts plotting to the groups selected for ANOVA (st.session_state.anova_groups)
    if that session variable exists.
    """
    attribute = st.session_state.anova_attribute
    p_value = anova.loc[metabolite, "p-corrected"]

    # create df that includes sample metadata and the metabolite column
    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)[[attribute, metabolite]].copy()

    # If the user selected a subset of groups for ANOVA, filter the boxplot to the same groups
    if "anova_groups" in st.session_state and st.session_state.anova_groups:
        df = df[df[attribute].isin(st.session_state.anova_groups)]

    # add sample id column for hover (index -> sample), rename to 'filename'
    df = df.reset_index().rename(columns={"index": "filename"})
    # if index had a name other than 'index' use that name
    if df.columns[0] == "filename" and st.session_state.data.index.name:
        df.rename(columns={"filename": st.session_state.data.index.name}, inplace=True)

    # pick hover cols
    hover_cols = []
    # always include filename (sample id) in hover
    hover_cols.append("filename")

    try:
        p_value_str = "{:.12g}".format(float(p_value))
    except Exception:
        p_value_str = str(p_value)

    title = f"{metabolite}<br>corrected p-value: {p_value_str}"
    fig = px.box(
        df,
        x=attribute,
        y=metabolite,
        template="plotly_white",
        width=800,
        height=600,
        points="all",
        color=attribute,
        hover_data=hover_cols,
    )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        xaxis_title=attribute,
        yaxis_title="intensity",
    )
    return fig


def gen_pairwise_tukey(df, metabolites, attribute):
    """Yield results for pairwise Tukey test for all metabolites between two options within the attribute."""
    for metabolite in metabolites:
        tukey = pg.pairwise_tukey(df, dv=metabolite, between=attribute)
        # tukey returns comparisons; we take the first row (0) because we filtered data earlier to only contain the two levels
        yield (
            metabolite,
            tukey.loc[0, "diff"],
            tukey.loc[0, "p-tukey"],
            attribute,
            tukey.loc[0, "A"],
            tukey.loc[0, "B"],
            tukey.loc[0, "mean(A)"],
            tukey.loc[0, "mean(B)"],
        )


def add_p_value_correction_to_tukeys(tukey, correction):
    if "p-corrected" not in tukey.columns:
        # add Bonferroni corrected p-values
        tukey.insert(
            3, "p-corrected", pg.multicomp(
                tukey["stats_p"].astype(float), method=correction)[1]
        )
        # add significance
        tukey.insert(4, "stats_significant", tukey["p-corrected"] < 0.05)
        # sort by p-value
        tukey.sort_values("stats_p", inplace=True)
    return tukey


@st.cache_data
def tukey(df, attribute, elements, correction):
    significant_metabolites = df.index  # NOTE: allow all metabolites here (user requested showing all boxplots/tukeys)
    data = pd.concat(
        [
            st.session_state.data.loc[:, significant_metabolites],
            st.session_state.md.loc[:, attribute],
        ],
        axis=1,
    )
    data = data[data[attribute].isin(elements)]
    tukey = pd.DataFrame(
        np.fromiter(
            gen_pairwise_tukey(data, significant_metabolites, attribute),
            dtype=[
                ("stats_metabolite", "U100"),
                (f"diff", "f"),
                ("stats_p", "f"),
                ("attribute", "U100"),
                ("A", "U100"),
                ("B", "U100"),
                ("mean(A)", "f"),
                ("mean(B)", "f"),
            ],
        )
    )
    tukey = tukey.dropna()
    tukey = add_p_value_correction_to_tukeys(tukey, correction)
    return tukey


def _get_tukey_feature_map(df_tukey):
    """Look up feature name mapping for stats_metabolite similar to anova."""
    return _get_feature_name_map()


@st.cache_resource
def get_tukey_teststat_plot(df):
    """Plot the test-statistic/diff (existing behaviour) but add hover text."""
    feature_map = _get_tukey_feature_map(df)
    fig = go.Figure()

    ins = df[df["stats_significant"] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=ins["diff"],
                y=-np.log(ins["stats_p"]),
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                # Add metabolite name or feature_ID as hover text
                hovertext=[
                    f"{m} — {feature_map[m]}" if feature_map and m in feature_map else str(m)
                    for m in ins["stats_metabolite"]
                ],
                hoverinfo="text",
            )
        )

    sig = df[df["stats_significant"] == True]
    if not sig.empty:
        fig.add_trace(
            go.Scatter(
                x=sig["diff"],
                y=-np.log(sig["stats_p"]),
                mode="markers+text",
                marker=dict(color="#ef553b"),
                text=["" for _ in sig["stats_metabolite"]],
                textposition="top right",
                textfont=dict(color="#ef553b", size=12),
                name="significant",
                hovertext=[
                    f"{m} — {feature_map[m]}" if feature_map and m in feature_map else str(m)
                    for m in sig["stats_metabolite"]
                ],
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


@st.cache_resource
def get_tukey_volcano_plot(df):
    """Volcano plot for Tukey: x = log2 fold change (mean(B)/mean(A)), y = -log10(p-value).
    Adds metabolite/feature hover labels.
    """
    feature_map = _get_tukey_feature_map(df)
    # compute log2 fold change (B relative to A). avoid zeros by a small epsilon
    eps = 1e-9
    meanA = df["mean(A)"].astype(float) + eps
    meanB = df["mean(B)"].astype(float) + eps
    df = df.copy()
    df["log2FC"] = np.log2(meanB / meanA)
    df["neglog10p"] = -np.log10(df["stats_p"].astype(float) + eps)

    fig = go.Figure()

    ins = df[df["stats_significant"] == False]
    if not ins.empty:
        fig.add_trace(
            go.Scatter(
                x=ins["log2FC"],
                y=ins["neglog10p"],
                mode="markers",
                marker=dict(color="#696880"),
                name="insignificant",
                # Add metabolite name or feature_ID as hover text
                hovertext=[
                    f"{m} — {feature_map[m]}" if feature_map and m in feature_map else str(m)
                    for m in ins["stats_metabolite"]
                ],
                hoverinfo="text",
            )
        )

    sig = df[df["stats_significant"] == True]
    if not sig.empty:
        fig.add_trace(
            go.Scatter(
                x=sig["log2FC"],
                y=sig["neglog10p"],
                mode="markers+text",
                marker=dict(color="#ef553b"),
                text=["" for _ in sig["stats_metabolite"]],
                textposition="top right",
                textfont=dict(color="#ef553b", size=12),
                name="significant",
                hovertext=[
                    f"{m} — {feature_map[m]}" if feature_map and m in feature_map else str(m)
                    for m in sig["stats_metabolite"]
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"TUKEY - {st.session_state.anova_attribute.upper()}: {st.session_state.tukey_elements[0]} - {st.session_state.tukey_elements[1]} (volcano)",
            "font_color": "#3E3D53",
        },
        xaxis_title="log2 Fold Change (mean B / mean A)",
        yaxis_title="-log10(p)",
        template="plotly_white",
    )
    return fig
