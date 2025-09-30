import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff

@st.cache_resource
def get_dendrogram(data, label_pos="bottom"):
    fig = ff.create_dendrogram(data, labels=list(data.index))
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(side=label_pos)
    return fig


@st.cache_resource
def get_heatmap(data, vmin=None, vmax=None):
    # SORT DATA TO CREATE HEATMAP

    # Compute linkage matrix from distances for hierarchical clustering
    linkage_data_ft = linkage(data, method="complete", metric="euclidean")
    linkage_data_samples = linkage(data.T, method="complete", metric="euclidean")

    # Create a dictionary of data structures computed to render the dendrogram.
    # We will use dict['leaves']
    cluster_samples = dendrogram(linkage_data_ft, no_plot=True)
    cluster_ft = dendrogram(linkage_data_samples, no_plot=True)

    # Create dataframe with sorted samples
    ord_samp = data.copy()
    ord_samp.reset_index(inplace=True)
    ord_samp = ord_samp.reindex(cluster_samples["leaves"])
    ord_samp.rename(columns={"index": "Filename"}, inplace=True)
    ord_samp.set_index("Filename", inplace=True)

    # Create dataframe with sorted features
    ord_ft = ord_samp.T.reset_index()
    ord_ft = ord_ft.reindex(cluster_ft["leaves"])
    # Set index to original metabolite names if available
    if "metabolite" in ord_ft.columns:
        ord_ft.set_index("metabolite", inplace=True)
    # Otherwise, keep the current index
    
    # Heatmap
    # Automatically set color range to 5th and 95th percentiles if not provided
    if vmin is None:
        vmin = np.nanpercentile(ord_ft.values, 5)
    if vmax is None:
        vmax = np.nanpercentile(ord_ft.values, 95)

    fig = px.imshow(
        ord_ft,
        y=ord_ft.index.tolist(),
        x=list(ord_ft.columns),
        text_auto=False,
        aspect="auto",
        color_continuous_scale="rainbow",
        zmin=vmin,
        zmax=vmax
    )

    fig.update_layout(
        autosize=False, width=700, height=1200, xaxis_title="", yaxis_title="",
    )

    # fig.update_yaxes(visible=False)
    fig.update_xaxes(tickangle=35)
    return fig, ord_ft
