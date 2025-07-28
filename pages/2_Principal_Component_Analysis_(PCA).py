import streamlit as st
from src.common import *
from src.pca import *

page_setup()

# pd.concat([st.session_state.md, st.session_state.data], axis=1)

st.markdown("# Principal Component Analysis (PCA)")

with st.expander("📖 About"):
    st.markdown(
        "Principal Component Analysis (PCA) is a statistical method used for dimensionality reduction in multivariate data analysis. It involves transforming a set of correlated variables into a smaller set of uncorrelated variables, known as principal components. These principal components are ordered by their ability to explain the variability in the data, with the first component accounting for the highest amount of variance. PCA can be used to simplify complex data sets, identify patterns and relationships among variables, and remove noise or redundancy from data."
    )
if not st.session_state.data.empty:

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
