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
            n_samps = len(committed_samps)
            cat_counts = md_all[md_all.index.isin(committed_samps)][attribute_col].value_counts()
            n_cats = len(cat_counts)
            if n_samps < 2:
                st.warning(f"⚠️ PCoA cannot be performed with fewer than 2 samples. Please adjust your filters to include more samples (currently {n_samps}).")
            elif cat_counts.min() < 2:
                st.warning("⚠️ Each category must have at least 2 samples. Please adjust your filters to include more samples.")
            else:
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

        # Only offer columns that have at least one non-NaN value among the filtered samples
        valid_color_cols = [
            col for col in st.session_state.md.columns
            if filtered_md[col].notna().any()
        ]
        if not valid_color_cols:
            valid_color_cols = list(st.session_state.md.columns)

        # Reset the color-by selection if it is no longer valid for the current filtered samples
        if st.session_state.get("pcoa_color_by") not in valid_color_cols:
            st.session_state["pcoa_color_by"] = valid_color_cols[0]

        col_dist, col_color = st.columns(2)
        with col_dist:
            st.selectbox(
                "Distance matrix",
                ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "hamming", "jaccard", "matching", "minkowski", "seuclidean", "sqeuclidean"],
                key="pcoa_distance_matrix",
                index = 6,
                help=(
                    "Euclidean is a good default. Bray-Curtis is meant for non-negative abundances "
                    "(avoid it after Center-Scaling, which creates negative values). Jaccard, Hamming and "
                    "Matching treat values as presence/absence or exact matches and are rarely meaningful "
                    "for continuous intensities. Minkowski uses p=2 (identical to Euclidean)."
                ),
            )
        with col_color:
            pcoa_color_by = st.selectbox(
                "Color by (also controls PERMANOVA grouping)",
                valid_color_cols,
                key="pcoa_color_by",
                help="Only metadata columns that have at least one value among the filtered samples are shown. The selected column is used to color the PCoA plot and as the grouping variable for PERMANOVA — independently of the filtering attribute above.",
            )

        # PERMANOVA uses the filtered samples, but grouped by the color-by attribute
        perm_md = filtered_md[filtered_md[pcoa_color_by].notna()]
        perm_data = filtered_data.loc[perm_md.index]
        n_unique = perm_md[pcoa_color_by].nunique()
        min_per_cat = perm_md[pcoa_color_by].value_counts().min() if n_unique > 0 else 0
        total_samples = len(filtered_md)

        if total_samples < 2:
            st.warning("⚠️ At least 2 samples are required to compute PCoA. Please adjust your filters.")
        else:
            can_permanova = n_unique >= 2 and min_per_cat >= 2

            if not can_permanova:
                if n_unique < 2:
                    st.warning(f"⚠️ PERMANOVA requires at least 2 categories in '{pcoa_color_by}' among the filtered samples — showing PCoA only.")
                elif min_per_cat < 2:
                    st.warning(f"⚠️ PERMANOVA requires at least 2 samples per category in '{pcoa_color_by}' among the filtered samples — showing PCoA only.")

            try:
                if can_permanova:
                    permanova, pcoa_result = permanova_pcoa(
                        perm_data,
                        st.session_state.pcoa_distance_matrix,
                        perm_md[pcoa_color_by],
                    )
                else:
                    permanova = None
                    pcoa_result = compute_pcoa_only(
                        filtered_data,
                        st.session_state.pcoa_distance_matrix,
                    )
            except Exception as e:
                st.error(
                    f"Could not compute the '{st.session_state.pcoa_distance_matrix}' distance matrix for this data "
                    f"(e.g., constant or all-zero samples/features produce undefined distances). "
                    f"Please choose a different distance metric. Details: {e}"
                )
                st.stop()

            # Dynamically determine available PCs from pcoa_result.samples columns
            available_pcs = [col for col in pcoa_result.samples.columns if col.startswith("PC")]
            available_pcs = available_pcs[:10]
            if len(available_pcs) < 2:
                st.warning("Not enough principal coordinates available for plotting.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    pcoa_x_axis = st.selectbox("Interested X-axis for plot", available_pcs, key="pcoa_x_axis")
                with col2:
                    pcoa_y_axis = st.selectbox("Interested Y-axis for plot", available_pcs, index=1 if len(available_pcs) > 1 else 0, key="pcoa_y_axis")

                if att_col == pcoa_color_by:
                    st.info("ℹ️ The **filter by** and **color by / PERMANOVA grouping** categories are the same — the plot will be organized by that single metadata category.")
                else:
                    st.info(f"ℹ️ The **filter by** (*{att_col}*) and **color by / PERMANOVA grouping** (*{pcoa_color_by}*) categories differ — points are filtered and shaped by *{att_col}*, but colored and grouped for PERMANOVA by *{pcoa_color_by}*.")

                if pcoa_x_axis == pcoa_y_axis:
                    st.warning("⚠️ X-axis and Y-axis cannot be the same. Please choose different axes to view results.")
                else:
                    shape_map = st.session_state.get("pcoa_committed_shapes", {})
                    if not shape_map:
                        shape_map = {str(cat): "circle" for cat in committed_categories}

                    def _render_pcoa_tabs(include_permanova):
                        if include_permanova:
                            t1, t2, t3, t4 = st.tabs(["📁 PERMANOVA statistics", "📈 Principal Coordinate Analysis", "📊 Explained variance", "📁 Data"])
                            with t1:
                                show_table(permanova, "PERMANOVA-statistics", hide_index=True)
                            pcoa_tab, var_tab, data_tab = t2, t3, t4
                        else:
                            pcoa_tab, var_tab, data_tab = st.tabs(["📈 Principal Coordinate Analysis", "📊 Explained variance", "📁 Data"])

                        with pcoa_tab:
                            # as in the protocol's plotPCoA(): show PERMDISP and PERMANOVA results on the plot
                            subtitle = None
                            if include_permanova:
                                _stats = dict(zip(permanova["Metric"], permanova["Value"]))
                                try:
                                    subtitle = f"PERMANOVA (p = {float(_stats['p-value']):.3g}, R² = {float(_stats['R2']):.4f})"
                                    if "PERMDISP p-value" in _stats:
                                        subtitle = f"PERMDISP p = {float(_stats['PERMDISP p-value']):.3g}; " + subtitle
                                except (KeyError, TypeError, ValueError):
                                    subtitle = None
                                if "PERMDISP p-value" in _stats and float(_stats["PERMDISP p-value"]) < 0.05:
                                    st.warning("⚠️ PERMDISP is significant (p < 0.05): group dispersions differ, which violates a PERMANOVA assumption. Interpret the PERMANOVA result with caution.")
                            fig = get_pcoa_scatter_plot(
                                pcoa_result,
                                st.session_state.md.loc[pcoa_result.samples.index],
                                pcoa_color_by,
                                pcoa_x_axis,
                                pcoa_y_axis,
                                shape_map=shape_map,
                                symbol_attribute=att_col,
                                subtitle=subtitle,
                            )
                            show_fig(fig, "principal-coordinate-analysis")
                            st.session_state["page_figs_pcoa_scatter"] = fig
                        with var_tab:
                            fig = get_pcoa_variance_plot(pcoa_result)
                            show_fig(fig, "pcoa-variance")
                            st.session_state["page_figs_pcoa_variance"] = fig
                        with data_tab:
                            show_table(pcoa_result.samples.iloc[:, :10], "principal-coordinates")

                    _render_pcoa_tabs(can_permanova and permanova is not None and not permanova.empty)

    else:
        st.warning("⚠️ Please complete data preparation step first!")

except ModuleNotFoundError:
    st.error("This page requires the `scikit-bio` (skbio) package, which is not installed. Install it with `pip install scikit-bio`.")
