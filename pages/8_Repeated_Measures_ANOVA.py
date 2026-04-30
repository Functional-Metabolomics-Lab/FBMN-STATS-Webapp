import streamlit as st
import pandas as pd
import time

from src.common import *
from src.repeated_anova import *

page_setup()
st.session_state["current_page"] = "Repeated Measures ANOVA"

st.markdown("# Repeated Measures ANOVA")

with st.expander("📖 About"):
    st.markdown(
        """
The **Repeated Measures ANOVA** tests whether the means of three or more related (within-subject) groups differ significantly.
Unlike the standard one-way ANOVA (which assumes independent samples), this test accounts for the fact that the same subjects
are measured under multiple conditions or time points.

##### 🧪 When to use it
- When the **same subjects** are measured across **three or more conditions** (e.g., timepoints, treatments).
- When your data are approximately **normally distributed** within each condition.
- As the parametric counterpart to the **Friedman test**.

##### ⚙️ Requirements
- A **subject identifier** column in your metadata that uniquely identifies each participant/sample across conditions.
- A **within-subject factor** column (e.g., timepoint, treatment) with at least 3 levels.
- Each subject must have exactly one observation per condition level.

##### 🧪 Choosing between tests
| Data structure | Parametric | Non-parametric |
|---|---|---|
| Two independent groups | T-test (Student / Welch) | Mann-Whitney U |
| Two paired groups | Paired T-test | Wilcoxon Signed-Rank |
| Three+ independent groups | One-way ANOVA | Kruskal-Wallis |
| **Three+ paired groups** | **Repeated Measures ANOVA** | Friedman |

##### 📊 Key outputs
- **F** – F-statistic measuring the ratio of between-condition variance to within-subject error variance.
- **p** – raw p-value for each metabolite.
- **p-corrected** – p-value adjusted for multiple comparisons (selected correction method).
- **significant** – whether p-corrected < 0.05.
        """
    )

# Ensure session state is initialized
st.session_state.setdefault("df_rm_anova", pd.DataFrame())

if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)

    # --- Within-subject factor selector ---
    prev_rm_anova_attribute = st.session_state.get("_prev_rm_anova_attribute", None)
    rm_anova_attribute = c1.selectbox(
        "Within-subject factor (condition/timepoint)",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="rm_anova_attribute",
    )

    if prev_rm_anova_attribute is not None and rm_anova_attribute != prev_rm_anova_attribute:
        st.session_state.df_rm_anova = pd.DataFrame()
    st.session_state["_prev_rm_anova_attribute"] = rm_anova_attribute

    attribute = st.session_state.rm_anova_attribute
    if attribute is None or attribute not in st.session_state.md.columns:
        st.warning("Please select a valid within-subject factor.")
        st.stop()

    # --- Subject column selector ---
    subject_options = [c for c in st.session_state.md.columns if c != attribute]
    prev_rm_subject = st.session_state.get("_prev_rm_anova_subject", None)
    rm_subject = c2.selectbox(
        "Subject identifier column",
        options=subject_options,
        key="rm_anova_subject",
        help="Column that uniquely identifies each subject/participant across conditions.",
    )

    if prev_rm_subject is not None and rm_subject != prev_rm_subject:
        st.session_state.df_rm_anova = pd.DataFrame()
    st.session_state["_prev_rm_anova_subject"] = rm_subject

    # --- Group selector ---
    attribute_options = list(set(st.session_state.md[attribute].dropna()))
    attribute_options.sort()

    prev_rm_anova_groups = st.session_state.get("_prev_rm_anova_groups", None)
    rm_anova_groups = c1.multiselect(
        "Select conditions to include (minimum 3)",
        options=attribute_options,
        default=attribute_options,
        key="rm_anova_groups",
        help="Select at least 3 levels of the within-subject factor.",
    )

    if prev_rm_anova_groups is not None and set(rm_anova_groups) != set(prev_rm_anova_groups):
        st.session_state.df_rm_anova = pd.DataFrame()
    st.session_state["_prev_rm_anova_groups"] = list(rm_anova_groups)

    min_required = 3
    run_disabled = not ("rm_anova_groups" in st.session_state and len(st.session_state.rm_anova_groups) >= min_required)

    st.button("Run Repeated Measures ANOVA", key="run_rm_anova", type="primary", disabled=run_disabled)

    color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
    st.selectbox("Color significant points by", options=color_by_options, key="rm_anova_color_by")

    if st.session_state.run_rm_anova:
        if "rm_anova_groups" not in st.session_state or len(st.session_state.rm_anova_groups) < min_required:
            st.error(f"At least {min_required} conditions must be selected.")
        else:
            progress_placeholder = st.empty()
            time_placeholder = st.empty()
            start_time = time.time()

            def progress_callback(done, total, est_left):
                progress = done / total
                progress_placeholder.progress(progress, text=f"Running RM ANOVA: metabolite {done} of {total}")
                time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")

            result = rm_anova_test(
                st.session_state.rm_anova_attribute,
                corrections_map[st.session_state.p_value_correction],
                elements=st.session_state.rm_anova_groups,
                subject_col=st.session_state.rm_anova_subject,
                _progress_callback=progress_callback,
            )
            progress_placeholder.empty()
            time_placeholder.empty()

            if result.empty:
                st.error(
                    "No results were returned. Ensure each subject has exactly one observation "
                    "per condition level and that you have selected a valid subject identifier."
                )
            else:
                st.session_state.df_rm_anova = result
                st.rerun()

    # --- Result tabs ---
    if st.session_state.df_rm_anova is not None and not st.session_state.df_rm_anova.empty:
        tab_labels = [
            "📈 RM ANOVA: plot",
            "📁 RM ANOVA: result table",
            "📊 RM ANOVA: metabolites (boxplots)",
        ]
        tabs = st.tabs(tab_labels)

        # --- Tab 0: Plot ---
        with tabs[0]:
            _rm_color = st.session_state.get("rm_anova_color_by", "Significance (default)")
            fig = get_rm_anova_plot(
                st.session_state.df_rm_anova,
                color_by=None if _rm_color == "Significance (default)" else _rm_color
            )
            show_fig(fig, "rm_anova")
            st.session_state["page_figs_rm_anova_plot"] = fig

        # --- Tab 1: Result table ---
        with tabs[1]:
            df_display = st.session_state.df_rm_anova.copy()

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
            all_metabolites = sorted(list(st.session_state.df_rm_anova["metabolite"]))

            def metabolite_label(m):
                return m.split("&")[0] if "&" in m else m

            st.selectbox(
                "select metabolite",
                all_metabolites,
                key="rm_anova_metabolite",
                format_func=metabolite_label,
            )

            met = st.session_state.rm_anova_metabolite
            if met in st.session_state.data.columns:
                fig = get_rm_anova_metabolite_boxplot(
                    st.session_state.df_rm_anova, met
                )
                show_fig(fig, f"rm_anova-{met}")
                st.session_state["page_figs_rm_anova_boxplot"] = fig
            else:
                st.warning("Selected metabolite not found in data columns.")

else:
    st.warning("⚠️ Please complete data preparation step first!")
