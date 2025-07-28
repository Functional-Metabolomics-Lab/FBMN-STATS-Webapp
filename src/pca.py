import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import numpy as np


@st.cache_data
def get_pca_df(scaled, n=5):
    # calculating Principal components
    pca = PCA(n_components=n)
    pca_df = pd.DataFrame(
        data=pca.fit_transform(scaled), 
        columns=[f"PC{x}" for x in range(1, n + 1)]
    )
    pca_df.index = scaled.index
    return pca.explained_variance_ratio_, pca_df


@st.cache_resource
def get_pca_scatter_plot(pca_df, pca_variance, attribute, md, pca_x_axis, pca_y_axis):
    title = f"PRINCIPAL COMPONENT ANALYSIS"

    df = pd.merge(
        pca_df[[pca_x_axis, pca_y_axis]],
        md[attribute].apply(str),
        left_index=True,
        right_index=True,
    )

    fig = px.scatter(
        df,
        x=pca_x_axis,
        y=pca_y_axis,
        template="plotly_white",
        width=600,
        height=400,
        color=attribute,
        hover_name=df.index,
    )

    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": title, "font_color": "#3E3D53"},
        xaxis_title=f"{pca_x_axis} ({round(pca_variance[int(pca_x_axis[2:]) - 1] * 100, 1)}%)",
        yaxis_title=f"{pca_y_axis} ({round(pca_variance[int(pca_y_axis[2:]) - 1] * 100, 1)}%)",
    )
    return fig

@st.cache_resource
def get_pca_scree_plot(pca_df, pca_variance):
    # To get a scree plot showing the variance of each PC in percentage:
    percent_variance = np.round(pca_variance * 100, decimals=2)

    fig = px.bar(
        x=pca_df.columns,
        y=percent_variance,
        template="plotly_white",
        width=500,
        height=400,
    )
    fig.update_traces(marker_color="#696880", width=0.5)
    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"text": "PCA - VARIANCE", "font_color": "#3E3D53"},
        xaxis_title="principal component",
        yaxis_title="variance (%)",
    )
    return fig
