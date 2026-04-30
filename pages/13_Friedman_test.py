import streamlit as st
import time
from src.common import *
from src.friedman import *

page_setup()
st.session_state["current_page"] = "Friedman"

st.markdown("# Friedman Test")

with st.expander("📖 About"):
    st.markdown(
        """
The **Friedman test** is a non-parametric alternative to **repeated-measures one-way ANOVA**.
It checks whether there are statistically significant differences **among three or more paired/matched groups** without assuming normal distribution or equal variances.

If the Friedman test indicates significant differences, you can apply a **pairwise Wilcoxon signed-rank post-hoc** test to directly compare a specific pair of groups and identify where those differences occur.

##### 🧪 When to use it
- When samples are **paired or matched** across three or more conditions (e.g., multiple time points on the same subjects, before / during / after treatment).
- When your data are **not normally distributed** or contain outliers.
- As a robust alternative to repeated-measures ANOVA.

> ⚠️ **Important:** This test requires that all groups contain the **same number of samples**, ordered to reflect the pairing (i.e., sample *i* in each group represents the same subject or matched observation). Groups are truncated to the shortest group length automatically.

##### 🧪 Choosing between tests
| Data structure | Parametric | Non-parametric |
|---|---|---|
| Two independent groups | T-test (Student / Welch) | Mann-Whitney U |
| Two paired groups | Paired T-test | Wilcoxon Signed-Rank |
| Three+ independent groups | One-way ANOVA | Kruskal-Wallis |
| Three+ paired groups | Repeated-measures ANOVA | **Friedman** |

##### 📊 Key outputs — Friedman
- **statistic** – Friedman chi-squared statistic.
- **p** – raw p-value.
- **p-corrected** – p-value adjusted for multiple comparisons (FDR).
- **significant** – whether p-corrected < 0.05.

##### 📊 Key outputs — Post-hoc (pairwise Wilcoxon)
- **W-val** – Wilcoxon W statistic for the pair.
- **p** – raw p-value from the pairwise Wilcoxon signed-rank test.
- **p-corrected** – adjusted p-value.
- **significant** – whether the pairwise comparison remains significant after correction.
        """
    )

if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)

    # --- Attribute selector ---
    prev_friedman_attribute = st.session_state.get("_prev_friedman_attribute", None)
    friedman_attribute = c1.selectbox(
        "attribute for Friedman test",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="friedman_attribute",
    )

    if prev_friedman_attribute is not None and friedman_attribute != prev_friedman_attribute:
        st.session_state.df_friedman = pd.DataFrame()
    st.session_state["_prev_friedman_attribute"] = friedman_attribute

    attribute = st.session_state.friedman_attribute
    if attribute is None or attribute not in st.session_state.md.columns:
        st.warning("Please select a valid attribute.")
        st.stop()

    attribute_options = list(set(st.session_state.md[attribute].dropna()))
    attribute_options.sort()

    # --- Group selector ---
    prev_friedman_groups = st.session_state.get("_prev_friedman_groups", None)
    friedman_groups = c2.multiselect(
        "Select groups to include in Friedman test (minimum 3)",
        options=attribute_options,
        default=attribute_options,
        key="friedman_groups",
        help="For comparing 2 paired groups, use the Wilcoxon Signed-Rank page instead. If button is disabled, select a different attribute.",
    )

    if prev_friedman_groups is not None and set(friedman_groups) != set(prev_friedman_groups):
        st.session_state.df_friedman = pd.DataFrame()
    st.session_state["_prev_friedman_groups"] = list(friedman_groups)

    min_required = 3
    run_disabled = not ("friedman_groups" in st.session_state and len(st.session_state.friedman_groups) >= min_required)

    st.button("Run Friedman test", key="run_friedman", type="primary", disabled=run_disabled)

    color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
    st.selectbox("Color significant points by", options=color_by_options, key="friedman_color_by")
    if st.session_state.run_friedman:
        if "friedman_groups" not in st.session_state or len(st.session_state.friedman_groups) < min_required:
            st.error(f"At least {min_required} groups must be selected to run the Friedman test.")
        else:
            progress_placeholder = st.empty()
            time_placeholder = st.empty()
            start_time = time.time()

            def progress_callback(done, total, est_left):
                progress = done / total
                elapsed = time.time() - start_time
                if done > 0:
                    est_total = elapsed / done * total
                    est_left = est_total - elapsed
                else:
                    est_left = 0
                progress_placeholder.progress(progress, text=f"Running Friedman test: metabolite {done} of {total}")
                time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")

            result = friedman_test(
                st.session_state.friedman_attribute,
                corrections_map[st.session_state.p_value_correction],
                elements=st.session_state.friedman_groups,
                _progress_callback=progress_callback,
            )
            progress_placeholder.empty()
            time_placeholder.empty()

            if result.empty:
                st.error(
                    "No results were returned. Ensure all groups have the **same number of samples** "
                    "(required for a paired design) and at least 2 observations each."
                )
            else:
                st.session_state.df_friedman = result
                st.rerun()

    # --- Tabs ---
    tab_options = [
        "📈 Friedman: plot",
        "📁 Friedman: result table",
        "📊 Friedman: metabolites (boxplots)",
    ]

    if st.session_state.df_friedman is not None and not st.session_state.df_friedman.empty:
        tabs = st.tabs(tab_options)

        # --- Tab 0: Friedman plot ---
        with tabs[0]:
            _friedman_color = st.session_state.get("friedman_color_by", "Significance (default)")
            fig = get_friedman_plot(st.session_state.df_friedman, color_by=None if _friedman_color == "Significance (default)" else _friedman_color)
            show_fig(fig, "friedman")
            st.session_state["page_figs_friedman_plot"] = fig

        # --- Tab 1: Friedman result table ---
        with tabs[1]:
            df_display = st.session_state.df_friedman.copy()

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
            for col in ["p", "p-corrected"]:
                if col in df_display.columns:
                    style_dict[col] = sci_notation_or_plain
            if style_dict:
                styled = df_display.style.format(style_dict)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_display, use_container_width=True, hide_index=True)

        # --- Tab 2: Metabolite boxplots ---
        with tabs[2]:
            all_metabolites = sorted(list(st.session_state.df_friedman["metabolite"]))

            def metabolite_label(m):
                return m.split("&")[0] if "&" in m else m

            st.selectbox(
                "select metabolite",
                all_metabolites,
                key="friedman_metabolite",
                format_func=metabolite_label,
            )

            met = st.session_state.friedman_metabolite
            if met in st.session_state.data.columns:
                fig = get_friedman_metabolite_boxplot(
                    st.session_state.df_friedman, met
                )
                show_fig(fig, f"friedman-{met}")
                st.session_state["page_figs_friedman_boxplot"] = fig
            else:
                st.warning("Selected metabolite not found in data columns.")

else:
    st.warning("⚠️ Please complete data preparation step first!")
