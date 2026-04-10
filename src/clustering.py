import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def parse_feature_labels(columns):
    """Parse feature labels formatted as ID_mz@RT.

    Returns two lists (mz_values, rt_values) aligned with the input columns.
    Entries that don't match the expected format are returned as None.
    """
    mz_values, rt_values = [], []
    for col in columns:
        try:
            at_idx = col.index("@")
            rt = float(col[at_idx + 1:])
            mz = float(col[:at_idx].rsplit("_", 1)[1])
            mz_values.append(mz)
            rt_values.append(rt)
        except (ValueError, IndexError):
            mz_values.append(None)
            rt_values.append(None)
    return mz_values, rt_values


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

    values = ord_ft.index.tolist()
    # Add dendrogram traces
    for trace in dendro['data']:
        fig.add_trace(trace, row=1, col=1)
    # Add heatmap trace(s)
    # Prepare row labels (split) and full names for hover
    row_labels = [y.split("&")[0] for y in ord_ft.index]
    full_names = list(ord_ft.index)
    fig.add_trace(
        go.Heatmap(
            z=ord_ft.values,
            x=list(ord_ft.columns),
            y=row_labels,
            colorscale=color,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title=""),
            name="",  # Hide trace name in hover
            customdata=np.array(full_names)[:, None].repeat(ord_ft.shape[1], axis=1),
            hovertemplate="Filename: %{x}<br>Metabolite&Name: %{customdata}<br>Abundance: %{z}<extra></extra>",
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

