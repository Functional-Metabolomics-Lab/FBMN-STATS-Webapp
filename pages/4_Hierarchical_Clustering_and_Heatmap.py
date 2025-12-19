import streamlit as st
from src.common import *
from src.clustering import *

page_setup("Hierarchical_Clustering_and_Heatmap")

st.markdown("# Hierarchical Clustering & Heatmap")

with st.expander("📖 About"):
    st.markdown(
    """
    **Hierarchical Clustering Analysis (HCA)** is an **unsupervised** method that groups samples based on their similarity.  
    Here, clustering is performed using a **Euclidean distance matrix** and **complete linkage** to merge clusters based on the maximum inter-cluster distance.  
    The result is displayed as a **dendrogram** (top), showing how closely samples or features cluster together.

    Below the dendrogram, a **heatmap** provides a color-coded view of feature intensities across samples, helping reveal patterns and co-varying features.  
    Similar samples or metabolites appear as blocks of similar colors, reflecting shared trends or functional relationships.

    All analyses use the same dataset provided in the “Data Preparation” stage.  
    Multiple **color-blind-friendly palettes** are available for the heatmap to enhance interpretability and accessibility.
    """
    )
    st.image("assets/figures/clustering.png")

if (hasattr(st.session_state, 'data') and st.session_state.data is not None and not st.session_state.data.empty):
    t1, t2 = st.tabs(["🧬 Clustered Heatmap", "📁 Heatmap Data"])
    with t1:
        st.info("Due to the large number of features, the row/column labels in the clustered heatmap may not always be fully visible or representative. For a more detailed view, please zoom in to explore the actual range of values. You can also hover over any row or feature in the heatmap to display the corresponding sample name and feature name. Double click to zoom out.")
        color = st.selectbox(
            "Select heatmap color palette",
            options=[
                'rainbow', 'viridis', 'cividis', 'plasma', 'inferno', 'magma',
                'gray', 'greys', 'blues', 'reds', 'greens', 'oranges', 'purples'
            ],
            index=0
        )
        st.session_state['heatmap_color'] = color
        fig, df = get_clustermap(st.session_state.data, st.session_state['heatmap_color'])
        st.session_state["cluster_fig"] = show_fig(fig, "clustermap")
    with t2:
        st.session_state["heatmap_data"] = show_table(df, "heatmap-data")
else:
    st.warning("⚠️ Please complete data preparation step first!")