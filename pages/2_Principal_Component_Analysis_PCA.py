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
    

    with c1:
        st.number_input(
            "number of components",
            2, 
            max_value = st.session_state.data.shape[0],
            value = 10,
            key = "n_components",
        )
        
    with c2: 
        st.selectbox(
            "attribute for PCA plot", 
            st.session_state.md.columns, 
            key="pca_attribute"
        )

    pca_variance, pca_df = get_pca_df(
        st.session_state.data, st.session_state.n_components
    )
    max_pca = min(10, pca_df.shape[1])
    pca_labels = [f"PC{i+1}" for i in range(max_pca)]

    with c1:
        pca_x_axis = st.selectbox(
            "Interested X-Axis", 
            pca_labels,
        )

    with c2: 
        pca_y_axis = st.selectbox(
            "Interested Y-Axis", 
            pca_labels, 
            index = 1
        )
        
    if pca_y_axis == pca_x_axis:
        st.warning("X-Axis and Y-Axis cannot be the same!")

    attribute_col = st.session_state.pca_attribute
    category_options = sorted(st.session_state.md[attribute_col].dropna().unique())
    
    st.multiselect(
        f"Filter categories in '{attribute_col}'",
        options=category_options,
        default=category_options,
        key="pca_category_filter"
    )

    md_filtered = st.session_state.md[st.session_state.md[attribute_col].isin(st.session_state.pca_category_filter)]
    pca_df_filtered = pca_df.loc[md_filtered.index]

    t1, t2, t3 = st.tabs(["📈 PCA Plot", "📊 Explained variance", "📁 Data"])
    with t1:
        fig = get_pca_scatter_plot(
            pca_df_filtered,
            pca_variance,
            st.session_state.pca_attribute,
            md_filtered, 
            pca_x_axis,
            pca_y_axis
        )
        show_fig(fig, "principal-component-analysis")
    with t2:
        fig = get_pca_scree_plot(pca_df, pca_variance)
        show_fig(fig, "pca-variance")
    with t3:
        show_table(pca_df_filtered, "principle-components")
else:
    st.warning("Please complete data preparation step first!")
