import streamlit as st
from src.common import *
from src.pca import *

page_setup("PCA")

# pd.concat([st.session_state.md, st.session_state.data], axis=1)

st.markdown("# Principal Component Analysis (PCA)")

with st.expander("📖 About"):
    st.markdown(
        "Principal Component Analysis (PCA) is an **unsupervised** dimensionality-reduction method used to explore patterns in multivariate data. "
    "It projects samples into a new coordinate space defined by *principal components (PCs)* — linear combinations of the original variables — "
    "that maximize the **Euclidean distance–based variance** between samples. "
    "The first few PCs (often the top 10) capture most of the variability in the dataset. "
    "In this app, you can choose any two of the top 10 PCs to visualize and examine how samples are distributed. "
    "Because PCA is unsupervised, observed clustering should be interpreted cautiously;"
    "it reflects variance in the data, not predefined group differences."
)
    
if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)

    # Set or update max components dynamically
    max_components = min(st.session_state.data.shape[0], st.session_state.data.shape[1])
    #st.write(f"Maximum number of components allowed: {max_components}")
   
    if "n_components" not in st.session_state:
        st.session_state["n_components"] = min(10, max_components)

    # Attribute selection
    with c1: 
        st.selectbox(
            "Attribute for PCA plot", 
            st.session_state.md.columns, 
            key="pca_attribute"
        )
    
    # Filter categories
    attribute_col = st.session_state.pca_attribute
    category_options = sorted(st.session_state.md[attribute_col].dropna().unique())

    with c2:
        st.multiselect(
            f"Filter categories in '{attribute_col}'",
            options=category_options,
            default=category_options,
            key="pca_category_filter"
        )

    # Filter metadata and data
    md_filtered = st.session_state.md[
        st.session_state.md[attribute_col].isin(st.session_state.pca_category_filter)
    ]
    
    data_filtered = st.session_state.data.loc[md_filtered.index]
   
    if data_filtered.shape[0] <= 2:
        st.warning("⚠️ PCA cannot be performed with fewer than 3 samples. Please adjust your filters to include more samples.")
    
    else:
        # Ensure n_components is valid after filtering
        max_allowed = min(data_filtered.shape[0], data_filtered.shape[1])
        n_components = min(10, max_allowed)
        st.session_state["n_components"] = n_components

        # Run PCA
        pca_variance, pca_df = get_pca_df(data_filtered, n_components)
        max_pca = min(10, pca_df.shape[1])
        pca_labels = [f"PC{i+1}" for i in range(max_pca)]

        # Axis selection
        col1, col2 = st.columns(2)
        with col1:
            pca_x_axis = st.selectbox("Interested X-Axis for the plot", pca_labels)

        with col2:
            pca_y_axis = st.selectbox("Interested Y-Axis for the plot", pca_labels, index=1)

        # Tabs for outputs
        if pca_x_axis == pca_y_axis:
            st.warning("⚠️ X-Axis and Y-Axis cannot be the same! Please choose different axes to view results.")
        else:
            t1, t2, t3 = st.tabs(["📈 PCA Scores Plot", "📊 Explained variance", "📁 Data"])
            with t1:
                fig = get_pca_scatter_plot(
                    pca_df,
                    pca_variance,
                    attribute_col,
                    md_filtered,
                    pca_x_axis,
                    pca_y_axis
                )
                show_fig(fig, "principal-component-analysis")

            with t2:
                fig = get_pca_scree_plot(pca_df, pca_variance)
                show_fig(fig, "pca-variance")

            with t3:
                show_table(pca_df, "principal-components")

else:
    st.warning("⚠️ Please complete the data preparation step first!")
