import streamlit as st
from src.common import *

PLOTLY_SHAPES = [
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "triangle-down", "star", "pentagon", "hexagon",
    "hexagram", "hourglass", "bowtie", "diamond-tall", "diamond-wide",
]

try:
    from src.pcoa import *

    @st.fragment
    def pcoa_advanced_filtering(attribute_col, all_categories, md_all):

        col1, col2 = st.columns(2)
        with col1:
            new_attr = st.selectbox(
                "Attribute for Filtering & Calculations",
                st.session_state.md.columns,
                key="pcoa_attribute",
            )

        # If the attribute changed inside the fragment, reset committed state and
        # trigger a full page rerun so all_categories / all_md are recomputed.
        if new_attr != attribute_col:
            new_cats = sorted(st.session_state.md[new_attr].dropna().unique())
            new_md = st.session_state.md[st.session_state.md[new_attr].notna()]
            st.session_state["pcoa_committed_categories"] = list(new_cats)
            st.session_state["pcoa_committed_samples"] = list(new_md.index)
            st.session_state["pcoa_filter_applied"] = False
            st.rerun()

        with col2:
            selected_cats = st.multiselect(
                f"Categories in '{attribute_col}'",
                options=all_categories,
                default=all_categories,
                key=f"pcoa_adv_categories_{attribute_col}",
            )

        if not selected_cats:
            st.warning("⚠️ At least one category must be selected.")

        selections = {}
        shape_selections = {}
        if selected_cats:
            header_cat, header_samp, header_shape = st.columns([1, 3, 1])
            header_cat.markdown("**Category**")
            header_samp.markdown("**Samples**")
            header_shape.markdown("**Shape**")
            committed_shapes = st.session_state.get("pcoa_committed_shapes", {})
            for ci, cat in enumerate(selected_cats):
                cat_samples = list(md_all[md_all[attribute_col] == cat].index)
                key = f"pcoa_adv_samples_{attribute_col}_{cat}"
                c_cat, c_samp, c_shape = st.columns([1, 3, 1])
                c_cat.write(str(cat))
                selections[cat] = c_samp.multiselect(
                    f"Samples for {cat}",
                    options=cat_samples,
                    default=cat_samples,
                    key=key,
                    label_visibility="collapsed",
                )
                default_shape = committed_shapes.get(str(cat), "circle")
                default_idx = PLOTLY_SHAPES.index(default_shape) if default_shape in PLOTLY_SHAPES else 0
                shape_selections[str(cat)] = c_shape.selectbox(
                    f"Shape for {cat}",
                    options=PLOTLY_SHAPES,
                    index=default_idx,
                    key=f"pcoa_adv_shape_{attribute_col}_{cat}",
                    label_visibility="collapsed",
                )

        if st.button("Done", type="primary", key="pcoa_adv_done", disabled=not selected_cats):
            committed_cats = selected_cats if selected_cats else all_categories
            committed_samps = []
            for s in selections.values():
                committed_samps.extend(s)
            if not committed_samps:
                committed_samps = list(md_all[md_all[attribute_col].isin(committed_cats)].index)
            st.session_state["pcoa_committed_categories"] = committed_cats
            st.session_state["pcoa_committed_samples"] = committed_samps
            st.session_state["pcoa_committed_shapes"] = dict(shape_selections)
            # Keep coloring aligned with the applied filtering attribute.
            st.session_state["pcoa_color_by"] = attribute_col
            st.session_state["pcoa_filter_applied"] = True
            st.rerun()

        # Compare current selections to committed to detect unsaved changes
        committed_cats = st.session_state.get("pcoa_committed_categories", all_categories)
        committed_samps = set(st.session_state.get("pcoa_committed_samples", list(md_all.index)))
        committed_shapes = st.session_state.get("pcoa_committed_shapes", {})
        current_samps = set(s for cat_samps in selections.values() for s in cat_samps)
        current_shapes = dict(shape_selections) if shape_selections else {}
        is_dirty = set(selected_cats) != set(committed_cats) or current_samps != committed_samps or current_shapes != committed_shapes

        if is_dirty:
            st.warning("⚠️ Unsaved changes — click Done to apply.")
        elif st.session_state.get("pcoa_filter_applied", False):
            n_cats = len(committed_cats)
            n_samps = len(committed_samps)
            st.success(f"✅ Filters applied! Showing {n_samps} sample(s) across {n_cats} categor{'y' if n_cats == 1 else 'ies'}.")

    page_setup()
    st.session_state["current_page"] = "PERMANOVA & PCoA"

    st.markdown("# Multivariate Statistics")
    st.markdown("### PERMANOVA & Principal Coordinate Analysis (PCoA)")

    with st.expander("📖 About"):
        st.markdown(
            """
            **Principal Coordinate Analysis (PCoA)** is an **unsupervised** ordination technique that visualizes relationships among samples based on a **distance (dissimilarity) matrix**. 
            Unlike PCA, which relies on Euclidean distance, PCoA can use **different distance metrics** such as Bray–Curtis, Jaccard, or Euclidean, making it suitable for non-normal or compositional data. 
            The method projects samples into new coordinate axes (PCo1, PCo2, etc.) that explain the greatest variation in distances. 
            Typically, the **first 10 coordinates** capture most of the variance, and users can select any two among these to visualize group separation.

            **PERMANOVA (Permutational Multivariate Analysis of Variance)** tests whether the **centroids of groups** differ significantly in multivariate space. 
            It uses **permutation-based resampling** (usually 999 permutations) to compute:
            - **Pseudo-F (test statistic):** measures the ratio of between-group to within-group variation  
            - **R²:** indicates the proportion of total variance explained by the grouping variable (metadata attribute)  
            - **p-value:** shows whether observed group differences are statistically significant under permutation  

            In this app, PERMANOVA results help quantify whether the separation observed in PCoA plots is statistically meaningful, rather than just visual. 
            """
)

        st.image("assets/figures/pcoa.png")


    if st.session_state.data is not None and not st.session_state.data.empty:
        # Initialize pcoa_attribute if not yet set
        if "pcoa_attribute" not in st.session_state:
            st.session_state["pcoa_attribute"] = st.session_state.md.columns[0]

        att_col = st.session_state.pcoa_attribute
        all_categories = sorted(st.session_state.md[att_col].dropna().unique())
        all_md = st.session_state.md[st.session_state.md[att_col].notna()]

        # Initialize committed state
        if "pcoa_committed_categories" not in st.session_state:
            st.session_state["pcoa_committed_categories"] = all_categories
        if "pcoa_committed_samples" not in st.session_state:
            st.session_state["pcoa_committed_samples"] = list(all_md.index)
        if "pcoa_color_by" not in st.session_state or st.session_state["pcoa_color_by"] not in st.session_state.md.columns:
            st.session_state["pcoa_color_by"] = att_col

        # Reset if committed categories are no longer valid (e.g. attribute changed)
        if not set(st.session_state["pcoa_committed_categories"]).issubset(set(all_categories)):
            st.session_state["pcoa_committed_categories"] = all_categories
            st.session_state["pcoa_committed_samples"] = list(all_md.index)

        with st.expander("Filtering Options"):
            pcoa_advanced_filtering(att_col, all_categories, all_md)
        committed_categories = st.session_state["pcoa_committed_categories"]
        committed_samples = st.session_state["pcoa_committed_samples"]
        committed_categories = [c for c in committed_categories if c in all_categories]
        committed_samples = [s for s in committed_samples if s in list(all_md.index)]
        if not committed_categories:
            committed_categories = all_categories
        if not committed_samples:
            committed_samples = list(all_md[all_md[att_col].isin(committed_categories)].index)

        filtered_md = all_md[all_md[att_col].isin(committed_categories)].loc[
            [s for s in committed_samples if s in all_md.index]
        ]
        filtered_data = st.session_state.data.loc[filtered_md.index]

        st.selectbox(
            "distance matrix",
            ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "hamming", "jaccard", "matching", "minkowski", "seuclidean", "sqeuclidean"],
            key="pcoa_distance_matrix",
            index = 6
        )

        # Display selected categories and their samples in a dataframe
        # st.markdown(f"**Selected categories in '{att_col}':** {', '.join(map(str, selected_categories))}")
        # st.dataframe(filtered_md[[att_col]], use_container_width=True)
        
        if len(committed_categories) < 2:
            st.warning("⚠️ PERMANOVA cannot be calculated for this group because there is only one category in the selected attribute. You need at least two categories to perform statistical testing.")
        elif filtered_md[st.session_state.pcoa_attribute].nunique() < 2:
            st.warning("⚠️ PERMANOVA cannot be calculated for this group because there is only one category in the selected attribute. You need at least two categories to perform statistical testing.")
        elif filtered_md[st.session_state.pcoa_attribute].value_counts().min() < 2:
            st.warning("⚠️ Each category must have at least 2 samples. Please adjust your filters to include more samples.")
        else:
            permanova, pcoa_result = permanova_pcoa(
                filtered_data,
                st.session_state.pcoa_distance_matrix,
                filtered_md[st.session_state.pcoa_attribute],
            )

            # Dynamically determine available PCs from pcoa_result.samples columns
            available_pcs = [col for col in pcoa_result.samples.columns if col.startswith("PC")]
            available_pcs = available_pcs[:10]
            if len(available_pcs) < 2:
                st.warning("Not enough principal coordinates available for plotting.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    pcoa_x_axis = st.selectbox("Interested X-axis for plot", available_pcs, key="pcoa_x_axis")
                with col2:
                    pcoa_y_axis = st.selectbox("Interested Y-axis for plot", available_pcs, index=1 if len(available_pcs) > 1 else 0, key="pcoa_y_axis")
                with col3:
                    pcoa_color_by = st.selectbox("Color by", st.session_state.md.columns, key="pcoa_color_by")

                if att_col == pcoa_color_by:
                    st.info("ℹ️ The **filter by** and **color by** categories are the same — the plot will be organized by that single metadata category.")
                else:
                    st.info(f"ℹ️ The **filter by** (*{att_col}*) and **color by** (*{pcoa_color_by}*) categories differ — points will be filtered and shaped by *{att_col}*, but colored by *{pcoa_color_by}*, creating subgroups.")

                if pcoa_x_axis == pcoa_y_axis:
                    st.warning("⚠️ X-axis and Y-axis cannot be the same. Please choose different axes to view results.")
                else:
                    if not permanova.empty:
                        t1, t2, t3, t4 = st.tabs(["📁 PERMANOVA statistics", "📈 Principal Coordinate Analysis", "📊 Explained variance", "📁 Data"])
                        with t1:
                            show_table(permanova, "PERMANOVA-statistics", hide_index=True)
                        with t2:
                            shape_map = st.session_state.get("pcoa_committed_shapes", {})
                            if not shape_map:
                                shape_map = {str(cat): "circle" for cat in committed_categories}
                            fig = get_pcoa_scatter_plot(
                                pcoa_result,
                                st.session_state.md.loc[filtered_md.index],
                                pcoa_color_by,
                                pcoa_x_axis,
                                pcoa_y_axis,
                                shape_map=shape_map,
                                symbol_attribute=att_col,
                            )
                            show_fig(fig, "principal-coordinate-analysis")
                            st.session_state["page_figs_pcoa_scatter"] = fig
                        with t3:
                            fig = get_pcoa_variance_plot(pcoa_result)
                            show_fig(fig, "pcoa-variance")
                            st.session_state["page_figs_pcoa_variance"] = fig
                        with t4:
                            filtered_samples = pcoa_result.samples.loc[filtered_md.index]
                            show_table(filtered_samples.iloc[:, :10], "principal-coordinates")

    else:
        st.warning("⚠️ Please complete data preparation step first!")

except ModuleNotFoundError:
    st.error("This page requires the `skbio` package, which is not available in the Windows app.")
