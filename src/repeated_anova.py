import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import time
from src.utils import get_feature_name_map


# ---------------------------------------------------------------------------
# Repeated Measures ANOVA
# ---------------------------------------------------------------------------

def gen_rm_anova_data(df_long, metabolites, subject_col, within_col, _progress_callback=None):
    """Yield (metabolite, p, F) for each metabolite using pg.rm_anova."""
    total = len(metabolites)
    start_time = time.time()
    for idx, col in enumerate(metabolites):
        subset = df_long[df_long["metabolite"] == col].dropna(subset=["value"])
        if subset.empty:
            continue
        try:
            result = pg.rm_anova(data=subset, dv="value", within=within_col, subject=subject_col)
        except Exception:
            if _progress_callback is not None:
                elapsed = time.time() - start_time
                done = idx + 1
                est_left = (elapsed / done) * (total - done) if done > 0 else 0
                _progress_callback(done, total, est_left)
            continue

        # Extract p and F from the result
        p = None
        f = None
        if "p-unc" in result.columns:
            p = float(result.iloc[0]["p-unc"])
        elif "p" in result.columns:
            p = float(result.iloc[0]["p"])
        if "F" in result.columns:
            f = float(result.iloc[0]["F"])

        if p is None or f is None:
            continue

        if _progress_callback is not None:
            elapsed = time.time() - start_time
            done = idx + 1
            est_left = (elapsed / done) * (total - done) if done > 0 else 0
            _progress_callback(done, total, est_left)

        yield col, p, f


def add_p_correction_to_rm_anova(df, correction):
    if "p-corrected" not in df.columns:
        df.insert(2, "p-corrected",
                  pg.multicomp(df["p"].astype(float), method=correction)[1])
    if "significant" not in df.columns:
        df.insert(3, "significant", df["p-corrected"] < 0.05)
    df.sort_values("p", inplace=True)
    return df


def rm_anova_test(attribute, correction, elements, subject_col, _progress_callback=None):
    """Run Repeated Measures ANOVA per metabolite.

    Parameters
    ----------
    attribute : str
        The within-subject factor column (from metadata).
    correction : str
        Multiple testing correction method.
    elements : list
        Levels of the within-subject factor to include.
    subject_col : str
        Column identifying the subject (from metadata).
    """
    combined = pd.concat([st.session_state.data, st.session_state.md], axis=1)
    if elements is not None:
        combined = combined[combined[attribute].isin(elements)]

    metabolite_cols = list(st.session_state.data.columns)

    # Melt to long form for pg.rm_anova
    id_vars = [subject_col, attribute]
    df_long = combined[id_vars + metabolite_cols].melt(
        id_vars=id_vars, var_name="metabolite", value_name="value"
    )

    # Drop rows where subject or value is missing
    df_long = df_long.dropna(subset=[subject_col, "value"])

    results = list(gen_rm_anova_data(df_long, metabolite_cols, subject_col, attribute,
                                     _progress_callback=_progress_callback))
    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results, columns=["metabolite", "p", "F"])
    df_res = df_res.dropna()
    if df_res.empty:
        return df_res
    df_res = add_p_correction_to_rm_anova(df_res, correction)
    return df_res


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Creating Repeated Measures ANOVA plot...")
def get_rm_anova_plot(rm_anova_df, color_by=None):
    """Scatter: x=log(F), y=-log(p)."""
    feature_map = get_feature_name_map()
    eps = 1e-12

    def make_hovertext(metabolites):
        htext = []
        for m in metabolites:
            met_name = feature_map[m] if feature_map and m in feature_map else str(m)
            htext.append(f"metabolite: {met_name}")
        return htext

    fig = go.Figure()

    total_points = len(rm_anova_df)
    n_significant = int(rm_anova_df["significant"].sum())
    n_insignificant = total_points - n_significant
    st.write(f"Significant: {n_significant}")
    st.write(f"Insignificant: {n_insignificant}")
    st.write(f"Total data points: {total_points}")

    ins = rm_anova_df[rm_anova_df["significant"] == False]
    if not ins.empty:
        fig.add_trace(go.Scatter(
            x=np.log(ins["F"].astype(float).clip(lower=eps)),
            y=-np.log(ins["p"].astype(float).clip(lower=eps)),
            mode="markers",
            marker=dict(color="#696880"),
            name="insignificant",
            hovertext=make_hovertext(ins["metabolite"]),
            hoverinfo="text",
        ))

    sig = rm_anova_df[rm_anova_df["significant"] == True]
    if not sig.empty:
        if color_by is not None:
            from src.utils import compute_dominant_groups
            dominant_map, all_groups = compute_dominant_groups(list(sig["metabolite"]), color_by)
            colors = px.colors.qualitative.Plotly
            for gi, group in enumerate(all_groups):
                group_sig = sig[sig["metabolite"].map(lambda m, g=group: dominant_map.get(m) == g)]
                if not group_sig.empty:
                    fig.add_trace(go.Scatter(
                        x=np.log(group_sig["F"].astype(float).clip(lower=eps)),
                        y=-np.log(group_sig["p"].astype(float).clip(lower=eps)),
                        mode="markers",
                        marker=dict(color=colors[gi % len(colors)]),
                        name=f"{group}",
                        hovertext=make_hovertext(group_sig["metabolite"]),
                        hoverinfo="text",
                    ))
        else:
            fig.add_trace(go.Scatter(
                x=np.log(sig["F"].astype(float).clip(lower=eps)),
                y=-np.log(sig["p"].astype(float).clip(lower=eps)),
                mode="markers",
                marker=dict(color="#ef553b"),
                name="significant",
                hovertext=make_hovertext(sig["metabolite"]),
                hoverinfo="text",
            ))

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={
            "text": f"Repeated Measures ANOVA - {st.session_state.rm_anova_attribute.upper()}",
            "font_color": "#3E3D53",
        },
        xaxis_title="log(F)",
        yaxis_title="-log(p)",
        showlegend=True,
        legend=dict(itemsizing='trace', font=dict(size=12), orientation="v"),
        template="plotly_white",
        width=600,
        height=600,
    )
    fig.update_yaxes(title_standoff=10)
    return fig


@st.cache_resource(show_spinner="Creating metabolite boxplot...")
def get_rm_anova_metabolite_boxplot(rm_anova_df, metabolite):
    """Boxplot for a single metabolite across within-subject groups."""
    attribute = st.session_state.rm_anova_attribute
    p_value = rm_anova_df.set_index("metabolite")._get_value(metabolite, "p-corrected")

    df = pd.concat([st.session_state.data, st.session_state.md], axis=1)[[attribute, metabolite]].copy()
    if "rm_anova_groups" in st.session_state and st.session_state.rm_anova_groups:
        df = df[df[attribute].isin(st.session_state.rm_anova_groups)]

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
