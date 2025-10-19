import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go


@st.cache_resource(show_spinner="Computing clustered heatmap...")
def get_clustermap(data, color, vmin=None, vmax=None, dendro_height=0.2, heatmap_height=0.75):
    # Compute linkage for clustering
    #linkage_data = linkage(data, method="complete", metric="euclidean")
    dendro = get_dendrogram(data, "bottom")
    #dendro.update_layout(width=700, height=300, margin=dict(l=0, r=0, t=0, b=0))
    dendro_leaves = dendro['layout']['xaxis']['ticktext']
    data_reordered = data.loc[dendro_leaves]
    # Create heatmap

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

    if vmin is None:
        vmin = np.nanpercentile(data_reordered.values, 5)
    if vmax is None:
        vmax = np.nanpercentile(data_reordered.values, 95)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[dendro_height, heatmap_height],
        shared_xaxes=True,
        vertical_spacing=0
    )

    # Add dendrogram traces
    for trace in dendro['data']:
        fig.add_trace(trace, row=1, col=1)
    # Add heatmap trace(s)
    fig.add_trace(
        go.Heatmap(
            z=ord_ft.values,
            x=list(ord_ft.columns),
            y=ord_ft.index.tolist(),
            colorscale=color,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title=""),
            name="",  # Hide trace name in hover
            hovertemplate="Filename: %{x}<br>Metabolite: %{y}<br>Abundance: %{z}<extra></extra>",
            # showscale=True
        ),
        row=2, col=1,
    )

    # st.plotly_chart(fig, use_container_width=True)
        
    # Update layout
    fig.update_layout(
        autosize=False, width=700, height=1200,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.update_xaxes(tickangle=35, row=2, col=1)
    return fig, ord_ft

@st.cache_resource
def get_dendrogram(data, label_pos="bottom"):
    fig = ff.create_dendrogram(data, labels=list(data.index))
    fig.update_layout()
    fig.update_xaxes(side=label_pos)
    return fig


# @st.cache_resource
# def get_heatmap(data, color, vmin=None, vmax=None):
#     # SORT DATA TO CREATE HEATMAP

#     # Compute linkage matrix from distances for hierarchical clustering
#     linkage_data_ft = linkage(data, method="complete", metric="euclidean")
#     linkage_data_samples = linkage(data.T, method="complete", metric="euclidean")

#     # Create a dictionary of data structures computed to render the dendrogram.
#     # We will use dict['leaves']
#     cluster_samples = dendrogram(linkage_data_ft, no_plot=True)
#     cluster_ft = dendrogram(linkage_data_samples, no_plot=True)

#     # Create dataframe with sorted samples
#     ord_samp = data.copy()
#     ord_samp.reset_index(inplace=True)
#     ord_samp = ord_samp.reindex(cluster_samples["leaves"])
#     ord_samp.rename(columns={"index": "Filename"}, inplace=True)
#     ord_samp.set_index("Filename", inplace=True)

#     # Create dataframe with sorted features
#     ord_ft = ord_samp.T.reset_index()
#     ord_ft = ord_ft.reindex(cluster_ft["leaves"])
#     # Set index to original metabolite names if available
#     if "metabolite" in ord_ft.columns:
#         ord_ft.set_index("metabolite", inplace=True)
#     # Otherwise, keep the current index
    
#     # Heatmap
#     if vmin is None:
#         vmin = np.nanpercentile(ord_ft.values, 5)
#     if vmax is None:
#         vmax = np.nanpercentile(ord_ft.values, 95)

#     fig = px.imshow(
#         ord_ft,
#         y=ord_ft.index.tolist(),
#         x=list(ord_ft.columns),
#         text_auto=False,
#         aspect="auto",
#         color_continuous_scale=color,
#         zmin=vmin,
#         zmax=vmax
#     )

#     fig.update_layout(
#         autosize=False, width=700, height=1200, xaxis_title="", yaxis_title="",
#     )

#     # fig.update_yaxes(visible=False)
#     fig.update_xaxes(tickangle=35)
#     return fig, ord_ft
