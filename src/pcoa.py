import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import skbio
from scipy.spatial import distance

@st.cache_data
def compute_pcoa_only(scaled, distance_metric):
    # distances are computed between samples (rows)
    distance_matrix = skbio.stats.distance.DistanceMatrix(
        distance.squareform(distance.pdist(scaled.values, distance_metric)),
        ids=scaled.index,
    )
    return skbio.stats.ordination.pcoa(distance_matrix)

@st.cache_data
def permanova_pcoa(scaled, distance_metric, attribute):
    
    # Create the distance matrix from the original data
    distance_matrix = skbio.stats.distance.DistanceMatrix(
        distance.squareform(distance.pdist(scaled.values, distance_metric)),
        ids = scaled.index, 
    )
    # perform PERMANOVA test
    permanova = skbio.stats.distance.permanova(distance_matrix, attribute)
    # R2 = SS_between / SS_total. With pseudo-F = (SS_between/(g-1)) / (SS_within/(n-g)):
    # R2 = F(g-1) / (F(g-1) + (n-g))
    f_stat = permanova["test statistic"]
    n_groups = permanova["number of groups"]
    n_samples = permanova["sample size"]
    permanova["R2"] = f_stat * (n_groups - 1) / (f_stat * (n_groups - 1) + (n_samples - n_groups))
    permanova_df = permanova.to_frame(name="PERMANOVA results").reset_index()
    permanova_df.columns = ["Metric", "Value"]

    # Homogeneity of group dispersions (PERMDISP), checked before PERMANOVA as in the protocol
    # (Step 36, vegan::betadisper + anova). A significant result (p < 0.05) means group dispersions
    # differ, so PERMANOVA results should be interpreted with caution.
    try:
        permdisp = skbio.stats.distance.permdisp(distance_matrix, attribute)
        permdisp_rows = pd.DataFrame({
            "Metric": ["PERMDISP test statistic (F)", "PERMDISP p-value"],
            "Value": [permdisp["test statistic"], permdisp["p-value"]],
        })
        permanova_df = pd.concat([permanova_df, permdisp_rows], ignore_index=True)
    except Exception:
        pass
    permanova_df["Value"] = permanova_df["Value"].apply(lambda x: str(x) if not isinstance(x, (int, float)) else x)
    # perfom PCoA
    pcoa = skbio.stats.ordination.pcoa(distance_matrix)
    
    return permanova_df, pcoa


# can not hash pcoa
def get_pcoa_scatter_plot(pcoa, md_samples, color_attribute, pcoa_x_axis, pcoa_y_axis, shape_map=None, symbol_attribute=None, subtitle=None):
    df = pcoa.samples[[pcoa_x_axis, pcoa_y_axis]]

    cols_to_merge = [color_attribute]
    if symbol_attribute and symbol_attribute != color_attribute:
        cols_to_merge.append(symbol_attribute)

    df = pd.merge(
        df[[pcoa_x_axis, pcoa_y_axis]],
        md_samples[cols_to_merge].apply(lambda c: c.apply(str)),
        left_index=True,
        right_index=True,
    )

    symbol_col = symbol_attribute if symbol_attribute else color_attribute

    title = f"PRINCIPAL COORDINATE ANALYSIS"
    if subtitle:
        title += f"<br><sup>{subtitle}</sup>"
    fig = px.scatter(
        df,
        x=pcoa_x_axis,
        y=pcoa_y_axis,
        template="plotly_white",
        width=600,
        height=400,
        color=color_attribute,
        symbol=symbol_col,
        symbol_map=shape_map if shape_map else None,
        hover_name=df.index,
    )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        xaxis_title=f"{pcoa_x_axis} ({round(pcoa.proportion_explained.iloc[int(pcoa_x_axis[2:]) - 1] * 100, 1)}%)",
        yaxis_title=f"{pcoa_y_axis} ({round(pcoa.proportion_explained.iloc[int(pcoa_y_axis[2:]) - 1] * 100, 1)}%)",
    )
    return fig

# can not hash pcoa
def get_pcoa_variance_plot(pcoa):
    # To get a scree plot showing the variance of each PC in percentage:
    percent_variance = np.round(pcoa.proportion_explained[:10] * 100, decimals=2)

    fig = px.bar(
        x=pcoa.samples.columns[:10],
        y=percent_variance,
        template="plotly_white",
        width=500,
        height=400,
    )
    fig.update_traces(marker_color="#696880", width=0.5)
    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": "PCoA - VARIANCE", "x": 0.5, "font_color": "#3E3D53"},
        xaxis_title="principal coordinate",
        yaxis_title="variance (%)",
    )
    return fig
