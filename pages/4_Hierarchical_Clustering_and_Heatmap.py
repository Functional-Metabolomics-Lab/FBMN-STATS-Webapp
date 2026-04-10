import streamlit as st
from src.common import *
from src.clustering import *

page_setup()
st.session_state["current_page"] = "Hierarchical Clustering & Heatmap"

@st.fragment
def hca_advanced_filtering(md_all):
    attribute_col = st.selectbox(
        "Attribute for filtering",
        md_all.columns,
        key="hca_attribute",
    )
    all_categories = sorted(md_all[attribute_col].dropna().unique())

    selected_cats = st.multiselect(
        f"Categories in '{attribute_col}'",
        options=all_categories,
        default=all_categories,
        key=f"hca_adv_categories_{attribute_col}",
    )

    if not selected_cats:
        st.warning("⚠️ At least one category must be selected.")

    selections = {}
    if selected_cats:
        header_cat, header_samp = st.columns([1, 4])
        header_cat.markdown("**Category**")
        header_samp.markdown("**Samples**")
        for cat in selected_cats:
            cat_samples = list(md_all[md_all[attribute_col] == cat].index)
            key = f"hca_adv_samples_{attribute_col}_{cat}"
            c_cat, c_samp = st.columns([1, 4])
            c_cat.write(str(cat))
            selections[cat] = c_samp.multiselect(
                f"Samples for {cat}",
                options=cat_samples,
                default=cat_samples,
                key=key,
                label_visibility="collapsed",
            )

    if st.button("Done", type="primary", key="hca_adv_done", disabled=not selected_cats):
        committed_cats = selected_cats if selected_cats else all_categories
        committed_samps = []
        for s in selections.values():
            committed_samps.extend(s)
        if not committed_samps:
            committed_samps = list(md_all[md_all[attribute_col].isin(committed_cats)].index)
        st.session_state["hca_committed_attribute"] = attribute_col
        st.session_state["hca_committed_categories"] = committed_cats
        st.session_state["hca_committed_samples"] = committed_samps
        st.session_state["hca_filter_applied"] = True
        st.rerun()

    committed_cats = st.session_state.get("hca_committed_categories", all_categories)
    committed_samps = set(st.session_state.get("hca_committed_samples", list(md_all.index)))
    current_samps = set(s for cat_samps in selections.values() for s in cat_samps)
    is_dirty = set(selected_cats) != set(committed_cats) or current_samps != committed_samps

    if selected_cats:
        if is_dirty:
            st.warning("⚠️ Unsaved changes — click Done to apply.")
        elif st.session_state.get("hca_filter_applied", False):
            n_cats = len(committed_cats)
            n_samps = len(committed_samps)
            st.success(f"✅ Filters applied! Showing {n_samps} sample(s) across {n_cats} categor{'y' if n_cats == 1 else 'ies'}.")

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

if st.session_state.data is not None and not st.session_state.data.empty:

    all_hca_md = st.session_state.md

    with st.expander("Advanced Filtering"):
        hca_advanced_filtering(all_hca_md)

    hca_attribute_col = st.session_state.get("hca_committed_attribute", st.session_state.md.columns[0])
    all_hca_categories = sorted(st.session_state.md[hca_attribute_col].dropna().unique())
    all_hca_md = st.session_state.md[st.session_state.md[hca_attribute_col].notna()]

    if "hca_committed_categories" not in st.session_state:
        st.session_state["hca_committed_categories"] = all_hca_categories
    if "hca_committed_samples" not in st.session_state:
        st.session_state["hca_committed_samples"] = list(all_hca_md.index)

    if not set(st.session_state["hca_committed_categories"]).issubset(set(all_hca_categories)):
        st.session_state["hca_committed_categories"] = all_hca_categories
        st.session_state["hca_committed_samples"] = list(all_hca_md.index)

    committed_categories = st.session_state["hca_committed_categories"]
    committed_samples = st.session_state["hca_committed_samples"]
    committed_categories = [c for c in committed_categories if c in all_hca_categories]
    committed_samples = [s for s in committed_samples if s in list(all_hca_md.index)]
    if not committed_categories:
        committed_categories = all_hca_categories
    if not committed_samples:
        committed_samples = list(all_hca_md[all_hca_md[hca_attribute_col].isin(committed_categories)].index)

    hca_md_filtered = all_hca_md[all_hca_md[hca_attribute_col].isin(committed_categories)].loc[
        [s for s in committed_samples if s in all_hca_md.index]
    ]
    hca_data_filtered = st.session_state.data.loc[hca_md_filtered.index]

    # Parse m/z and RT values from feature labels (format: ID_mz@RT)
    all_cols = list(hca_data_filtered.columns)
    mz_vals, rt_vals = parse_feature_labels(all_cols)

    valid_mz = [v for v in mz_vals if v is not None]
    valid_rt = [v for v in rt_vals if v is not None]

    mz_min = float(min(valid_mz)) if valid_mz else 0.0
    mz_max = float(max(valid_mz)) if valid_mz else 1000.0
    rt_min = float(min(valid_rt)) if valid_rt else 0.0
    rt_max = float(max(valid_rt)) if valid_rt else 100.0

    # Feature filter controls
    col_name, col_mz_min, col_mz_max, col_rt_min, col_rt_max = st.columns(5)

    name_query = col_name.text_input(
        "Name", placeholder="e.g. 1234",
        help="Filter by feature name (case-insensitive, partial match)"
    )
    mz_min_input = col_mz_min.number_input(
        "m/z min", value=None, min_value=mz_min, max_value=mz_max,
        placeholder=f"{mz_min:.4f}", format="%.4f",
        help=f"Dataset range: {mz_min:.4f} – {mz_max:.4f}"
    )
    mz_max_input = col_mz_max.number_input(
        "m/z max", value=None, min_value=mz_min, max_value=mz_max,
        placeholder=f"{mz_max:.4f}", format="%.4f",
        help=f"Dataset range: {mz_min:.4f} – {mz_max:.4f}"
    )
    rt_min_input = col_rt_min.number_input(
        "RT min", value=None, min_value=rt_min, max_value=rt_max,
        placeholder=f"{rt_min:.2f}", format="%.2f",
        help=f"Dataset range: {rt_min:.2f} – {rt_max:.2f}"
    )
    rt_max_input = col_rt_max.number_input(
        "RT max", value=None, min_value=rt_min, max_value=rt_max,
        placeholder=f"{rt_max:.2f}", format="%.2f",
        help=f"Dataset range: {rt_min:.2f} – {rt_max:.2f}"
    )

    # Apply filters — only active fields are used; all conditions combined with AND
    mask = [True] * len(all_cols)

    if name_query.strip():
        q = name_query.strip().lower()
        mask = [m and q in col.lower() for m, col in zip(mask, all_cols)]
        if not any(mask):
            st.warning(f"⚠️ No features found matching name '{name_query}'.")

    if mz_min_input is not None:
        mask = [m and (v is not None and v >= mz_min_input) for m, v in zip(mask, mz_vals)]
    if mz_max_input is not None:
        mask = [m and (v is not None and v <= mz_max_input) for m, v in zip(mask, mz_vals)]
    if rt_min_input is not None:
        mask = [m and (v is not None and v >= rt_min_input) for m, v in zip(mask, rt_vals)]
    if rt_max_input is not None:
        mask = [m and (v is not None and v <= rt_max_input) for m, v in zip(mask, rt_vals)]

    filtered_cols = [col for col, keep in zip(all_cols, mask) if keep]

    if not filtered_cols:
        st.warning("⚠️ No features match the current filters. Please adjust your criteria.")
        st.stop()

    data_for_heatmap = hca_data_filtered[filtered_cols]
    st.caption(f"Showing **{len(filtered_cols)}** of **{len(all_cols)}** features.")

    n_samples, n_features = data_for_heatmap.shape
    if n_samples < 2:
        st.warning("⚠️ Hierarchical clustering requires at least 2 samples. Please adjust your filters.")
        st.stop()
    if n_features < 2:
        st.warning("⚠️ Hierarchical clustering requires at least 2 features. Please adjust your filters.")
        st.stop()

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
        with st.spinner("Creating heatmap..."):
            fig, df = get_clustermap(data_for_heatmap, st.session_state['heatmap_color'])
        st.session_state["cluster_fig"] = show_fig(fig, "clustermap")
    with t2:
        st.session_state["heatmap_data"] = show_table(df, "heatmap-data")
else:
    st.warning("⚠️ Please complete data preparation step first!")