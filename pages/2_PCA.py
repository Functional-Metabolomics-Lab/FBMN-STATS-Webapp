import streamlit as st
from src.common import *
from src.pca import *

page_setup()
st.session_state["current_page"] = "PCA"

# pd.concat([st.session_state.md, st.session_state.data], axis=1)

@st.fragment
def advanced_filtering(attribute_col, all_categories, md_all):

    col1, col2 = st.columns(2)
    with col1:
        new_attr = st.selectbox(
                    "Attribute for Filtering & Calculations",
                    st.session_state.md.columns,
                    key="pca_attribute"
                )

    # If the attribute changed inside the fragment, reset committed state and
    # trigger a full page rerun so all_categories / all_md are recomputed.
    if new_attr != attribute_col:
        new_cats = sorted(st.session_state.md[new_attr].dropna().unique())
        new_md = st.session_state.md[st.session_state.md[new_attr].notna()]
        st.session_state["pca_committed_categories"] = list(new_cats)
        st.session_state["pca_committed_samples"] = list(new_md.index)
        st.session_state["pca_filter_applied"] = False
        st.rerun()

    with col2:
        selected_cats = st.multiselect(
            f"Categories in '{attribute_col}'",
            options=all_categories,
            default=all_categories,
            key=f"adv_categories_{attribute_col}",
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
            key = f"adv_samples_{attribute_col}_{cat}"
            c_cat, c_samp = st.columns([1, 4])
            c_cat.write(str(cat))
            selections[cat] = c_samp.multiselect(
                f"Samples for {cat}",
                options=cat_samples,
                default=cat_samples,
                key=key,
                label_visibility="collapsed",
            )

    if st.button("Done", type="primary", disabled=not selected_cats):
        committed_cats = selected_cats if selected_cats else all_categories
        committed_samps = []
        for s in selections.values():
            committed_samps.extend(s)
        if not committed_samps:
            committed_samps = list(md_all[md_all[attribute_col].isin(committed_cats)].index)
        st.session_state["pca_committed_categories"] = committed_cats
        st.session_state["pca_committed_samples"] = committed_samps
        st.session_state["pca_filter_applied"] = True
        st.rerun()

    # Compare current selections to committed to detect unsaved changes
    committed_cats = st.session_state.get("pca_committed_categories", all_categories)
    committed_samps = set(st.session_state.get("pca_committed_samples", list(md_all.index)))
    current_samps = set(s for cat_samps in selections.values() for s in cat_samps)
    is_dirty = set(selected_cats) != set(committed_cats) or current_samps != committed_samps

    if is_dirty:
        st.warning("⚠️ Unsaved changes - click Done to apply.")
    elif st.session_state.get("pca_filter_applied", False):
        n_cats = len(committed_cats)
        n_samps = len(committed_samps)
        st.success(f"✅ Filters applied! Showing {n_samps} sample(s) across {n_cats} categor{'y' if n_cats == 1 else 'ies'}.")

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
    # Initialize pca_attribute if not yet set
    if "pca_attribute" not in st.session_state:
        st.session_state["pca_attribute"] = st.session_state.md.columns[0]

    attribute_col = st.session_state.pca_attribute
    all_categories = sorted(st.session_state.md[attribute_col].dropna().unique())
    all_md = st.session_state.md[st.session_state.md[attribute_col].notna()]

    # Initialize committed state
    if "pca_committed_categories" not in st.session_state:
        st.session_state["pca_committed_categories"] = all_categories
    if "pca_committed_samples" not in st.session_state:
        st.session_state["pca_committed_samples"] = list(all_md.index)

    # Reset if committed categories are no longer valid (e.g. attribute changed)
    if not set(st.session_state["pca_committed_categories"]).issubset(set(all_categories)):
        st.session_state["pca_committed_categories"] = all_categories
        st.session_state["pca_committed_samples"] = list(all_md.index)

    with st.expander("Filtering Options"):
        advanced_filtering(attribute_col, all_categories, all_md)

    committed_categories = st.session_state["pca_committed_categories"]
    committed_samples = st.session_state["pca_committed_samples"]
    committed_categories = [c for c in committed_categories if c in all_categories]
    committed_samples = [s for s in committed_samples if s in list(all_md.index)]
    if not committed_categories:
        committed_categories = all_categories
    if not committed_samples:
        committed_samples = list(all_md[all_md[attribute_col].isin(committed_categories)].index)

    md_filtered = all_md[all_md[attribute_col].isin(committed_categories)].loc[
        [s for s in committed_samples if s in all_md.index]
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

        # Axis selection and color option
        col1, col2, col3 = st.columns(3)
        with col1:
            pca_x_axis = st.selectbox("Interested X-Axis", pca_labels)

        with col2:
            pca_y_axis = st.selectbox("Interested Y-Axis", pca_labels, index=1)

        with col3:
            pca_color_by = st.selectbox("Color by", st.session_state.md.columns, key="pca_color_by")

        # Tabs for outputs
        if pca_x_axis == pca_y_axis:
            st.warning("⚠️ X-Axis and Y-Axis cannot be the same! Please choose different axes to view results.")
        else:
            t1, t2, t3 = st.tabs(["📈 PCA Scores Plot", "📊 Explained variance", "📁 Data"])
            with t1:
                fig = get_pca_scatter_plot(
                    pca_df,
                    pca_variance,
                    pca_color_by,
                    st.session_state.md.loc[md_filtered.index],
                    pca_x_axis,
                    pca_y_axis
                )
                show_fig(fig, "principal-component-analysis")
                st.session_state["page_figs_pca_scores"] = fig

            with t2:
                fig = get_pca_scree_plot(pca_df, pca_variance)
                show_fig(fig, "pca-variance")
                st.session_state["page_figs_pca_scree"] = fig

            with t3:
                show_table(pca_df, "principal-components")

else:
    st.warning("⚠️ Please complete the data preparation step first!")
