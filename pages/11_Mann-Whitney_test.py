import streamlit as st
import pandas as pd
from src.common import *
from src.mannwhit import *

page_setup()
st.session_state["current_page"] = "Mann-Whitney U"

st.markdown("# Mann-Whitney U Test (MWU)")

with st.expander("📖 About"):
    st.markdown("""
The Mann-Whitney U test (also called Wilcoxon rank-sum test) is a non-parametric test for comparing the distributions of two independent groups. It does not assume normality and is suitable for continuous or ordinal data.

##### 🧪 When to use MWU
- When your data are not normally distributed or are ordinal.
- When comparing two independent groups (not paired).

##### ⚙️ Alternative hypotheses
- **Two-sided (default):** tests for any difference in distributions.
- **Greater:** tests if group A > group B.
- **Less:** tests if group A < group B.

##### 📊 Key outputs
- **U** – Mann-Whitney U statistic.
- **p-val** – probability that the observed difference is due to chance (p < 0.05 = significant).
- **p-corrected (FDR)** – adjusted p-values for multiple comparisons.
- **Significance** – marks whether the adjusted result remains significant (after FDR).
    """)

# Ensure st.session_state.df_mwu is initialized
if "df_mwu" not in st.session_state:
    st.session_state.df_mwu = pd.DataFrame()

if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)
    def clear_mwu_data():
        st.session_state.df_mwu = pd.DataFrame()
        st.session_state.pop("mwu_attempted_metabolites", None)
        st.session_state.pop("mwu_returned_metabolites", None)
        for key in list(st.session_state.keys()):
            if key.startswith("mwu_metabolite") or key.startswith("_page_tab_"):
                del st.session_state[key]

    c1.selectbox(
        "select attribute of interest",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="mwu_attribute",
        on_change=clear_mwu_data
    )

    if st.session_state.mwu_attribute is not None:
        attribute_options = list(
            set(st.session_state.md[st.session_state.mwu_attribute].dropna())
        )
        attribute_options.sort()
    else:
        attribute_options = []
    c2.multiselect(
        "select **two** options from the attribute for comparison",
        options=attribute_options,
        default=attribute_options[:2],
        key="mwu_options",
        max_selections=2,
        help="Select two options.",
        on_change=clear_mwu_data
    )
    c1, c2 = st.columns(2)
    v_space(2, c1)
    c2.selectbox(
        "alternative",
        options=["two-sided", "greater", "less"],
        key="mwu_alternative",
        help="Choose the test direction: 'two-sided' checks for any difference; 'greater' tests if group A > group B; and 'less' tests if group A < group B.",
        on_change=clear_mwu_data
    )

    if c1.button("Run Mann-Whitney U test", type="primary", disabled=(len(st.session_state.mwu_options) != 2)):
        progress_placeholder = st.empty()
        time_placeholder = st.empty()
        def progress_callback(done, total, est_left):
            progress = done / total
            progress_placeholder.progress(progress, text=f"Running Mann-Whitney U test: metabolite {done} of {total}")
            time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")

        st.session_state.df_mwu = gen_mwu_data(
            st.session_state.mwu_attribute,
            st.session_state.mwu_options,
            st.session_state.mwu_alternative,
            corrections_map[st.session_state.p_value_correction],
            _progress_callback=progress_callback
        )
        progress_placeholder.empty()
        time_placeholder.empty()
        st.rerun()

    mwu_attempted = st.session_state.get("mwu_attempted_metabolites")
    mwu_returned = st.session_state.get("mwu_returned_metabolites")
    if mwu_attempted is not None and mwu_returned is not None and mwu_attempted != mwu_returned:
        st.warning(
            f"Mann-Whitney U attempted {mwu_attempted} metabolites, but only {mwu_returned} produced plottable results. "
            f"{mwu_attempted - mwu_returned} metabolite(s) were skipped because valid test outputs could not be computed for the selected filters."
        )

    df = st.session_state.df_mwu
    if df is not None and not df.empty:
        tabs = st.tabs(["📈 Feature significance", "📊 Single metabolite plots", "📁 Data"])

        with tabs[0]:
            color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
            st.selectbox("Color significant points by", options=color_by_options, key="mwu_color_by")
            _mwu_color = st.session_state.get("mwu_color_by", "Significance (default)")
            mwu_plot_df = filter_top_significant_points_ui(df, "mwu_plot")
            fig = plot_mwu(mwu_plot_df, color_by=None if _mwu_color == "Significance (default)" else _mwu_color)
            show_fig(fig, "mwu")
            st.session_state["page_figs_mwu_sig"] = fig

        with tabs[1]:
            metabolite_options = list(df.index)
            def metabolite_label(m):
                return m.split("&")[0] if "&" in m else m
            st.selectbox("metabolite", metabolite_options, key="mwu_metabolite", format_func=metabolite_label)
            if st.session_state.mwu_metabolite in st.session_state.data.columns:
                fig = mwu_boxplot(df, st.session_state.mwu_metabolite)
                if fig is not None:
                    show_fig(fig, f"mwu-boxplot-{st.session_state.mwu_metabolite}", container_width=True)
                    st.session_state["page_figs_mwu_boxplot"] = fig
            else:
                st.warning(f"Selected metabolite not found in data columns. Please select a valid metabolite.")

        with tabs[2]:
            df_display = df.copy()
            def sci_notation_or_plain(x):
                try:
                    if pd.isnull(x):
                        return x
                    if float(x) == 0:
                        return 0
                    return f"{x:.2e}"
                except Exception:
                    return x
            style_dict = {}
            for col in ["p-val", "p-corrected"]:
                if col in df_display.columns:
                    style_dict[col] = sci_notation_or_plain
            if style_dict:
                styled = df_display.style.format(style_dict)
                st.dataframe(styled, use_container_width=True)
            else:
                st.dataframe(df_display, use_container_width=True)
else:
    st.warning("⚠️ Please complete data preparation step first!")
