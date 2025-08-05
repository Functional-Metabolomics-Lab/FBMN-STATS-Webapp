import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import skbio
from scipy.spatial import distance

@st.cache_data
def permanova_pcoa(scaled, distance_matrix, attribute):
    # Create the distance matrix from the original data
    distance_matrix = skbio.stats.distance.DistanceMatrix(
        distance.squareform(distance.pdist(scaled.values, distance_matrix)),
        ids = scaled.index, 
    )
    # perform PERMANOVA test
    permanova = skbio.stats.distance.permanova(distance_matrix, attribute)
    permanova["R2"] = 1 - 1 / (1 + permanova["test statistic"] * permanova["number of groups"] / (permanova["sample size"] - permanova["number of groups"] - 1))
    # perfom PCoA
    pcoa = skbio.stats.ordination.pcoa(distance_matrix)
    
    return permanova, pcoa


# can not hash pcoa
def get_pcoa_scatter_plot(pcoa, md_samples, attribute, pcoa_x_axis, pcoa_y_axis, allowed_categories):
    df = pcoa.samples[[pcoa_x_axis, pcoa_y_axis]]
    df = df.set_index(md_samples.index)
    df = pd.merge(
        df[[pcoa_x_axis, pcoa_y_axis]],
        md_samples[attribute].apply(str),
        left_index=True,
        right_index=True,
    )

    df = df[df[attribute].isin(allowed_categories)]

    title = f"PRINCIPAL COORDINATE ANALYSIS"
    fig = px.scatter(
        df,
        x=pcoa_x_axis,
        y=pcoa_y_axis,
        template="plotly_white",
        width=600,
        height=400,
        color=attribute,
        hover_name=df.index,
    )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        xaxis_title=f"{pcoa_x_axis} ({round(pcoa.proportion_explained[int(pcoa_x_axis[2:]) - 1] * 100, 1)}%)",
        yaxis_title=f"{pcoa_y_axis} ({round(pcoa.proportion_explained[int(pcoa_y_axis[2:]) - 1] * 100, 1)}%)",
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
        xaxis_title="principal component",
        yaxis_title="variance (%)",
    )
    return fig
