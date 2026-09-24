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

    # Hierarchical clustering of the features (rows of the heatmap); the sample order
    # (heatmap columns) is taken from the dendrogram above.
    linkage_features = linkage(data.T, method="complete", metric="euclidean")
    cluster_ft = dendrogram(linkage_features, no_plot=True)

    # features x samples, features reordered by their clustering
    ord_ft = data.T.iloc[cluster_ft["leaves"]]
    ord_ft.columns.name = "Filename"

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

    # Align heatmap columns with the dendrogram leaves: the dendrogram draws its leaves at
    # numeric x positions (tickvals) in the order given by ticktext, so place the heatmap
    # columns at exactly those positions.
    leaf_labels = list(dendro['layout']['xaxis']['ticktext'])
    leaf_positions = list(dendro['layout']['xaxis']['tickvals'])
    ord_ft = ord_ft[leaf_labels]

    # Add dendrogram traces
    for trace in dendro['data']:
        fig.add_trace(trace, row=1, col=1)
    # Add heatmap trace(s)
    # Prepare row labels (split) and full names for hover
    row_labels = [str(y).split("&")[0] for y in ord_ft.index]
    full_names = [str(y) for y in ord_ft.index]
    hover = np.array([[f"Filename: {s}<br>Metabolite&Name: {m}" for s in leaf_labels] for m in full_names])
    fig.add_trace(
        go.Heatmap(
            z=ord_ft.values,
            x=leaf_positions,
            y=row_labels,
            colorscale=color,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="", len=heatmap_height, y=0, yanchor="bottom"),
            name="",  # Hide trace name in hover
            customdata=hover,
            hovertemplate="%{customdata}<br>Abundance: %{z}<extra></extra>",
            # showscale=True
        ),
        row=2, col=1,
    )
    fig.update_xaxes(tickmode="array", tickvals=leaf_positions, ticktext=leaf_labels, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

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

