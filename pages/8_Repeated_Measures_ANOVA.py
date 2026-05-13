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
        st.session_state.pop("rm_anova_attempted_metabolites", None)
        st.session_state.pop("rm_anova_returned_metabolites", None)
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
        st.session_state.pop("rm_anova_attempted_metabolites", None)
        st.session_state.pop("rm_anova_returned_metabolites", None)
    st.session_state["_prev_rm_anova_subject"] = rm_subject

    # --- Group selector ---
    attribute_options = list(set(st.session_state.md[attribute].dropna()))
    attribute_options.sort()

    prev_rm_anova_groups = st.session_state.get("_prev_rm_anova_groups", None)
    rm_anova_groups = c1.multiselect(
        "Select groups to include (minimum 3)",
        options=attribute_options,
        default=attribute_options,
        key="rm_anova_groups",
        help="Select at least 3 levels of the within-subject factor.",
    )

    if prev_rm_anova_groups is not None and set(rm_anova_groups) != set(prev_rm_anova_groups):
        st.session_state.df_rm_anova = pd.DataFrame()
        st.session_state.pop("rm_anova_attempted_metabolites", None)
        st.session_state.pop("rm_anova_returned_metabolites", None)
    st.session_state["_prev_rm_anova_groups"] = list(rm_anova_groups)

    min_required = 3
    run_disabled = not ("rm_anova_groups" in st.session_state and len(st.session_state.rm_anova_groups) >= min_required)

    st.button("Run Repeated Measures ANOVA", key="run_rm_anova", type="primary", disabled=run_disabled)

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

    rm_attempted = st.session_state.get("rm_anova_attempted_metabolites")
    rm_returned = st.session_state.get("rm_anova_returned_metabolites")
    if rm_attempted is not None and rm_returned is not None and rm_attempted != rm_returned:
        st.warning(
            f"Repeated Measures ANOVA attempted {rm_attempted} metabolites, but only {rm_returned} produced plottable results. "
            f"{rm_attempted - rm_returned} metabolite(s) were skipped because valid test outputs could not be computed for the selected filters."
        )

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
            color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
            st.selectbox("Color significant points by", options=color_by_options, key="rm_anova_color_by")
            _rm_color = st.session_state.get("rm_anova_color_by", "Significance (default)")
            rm_anova_plot_df = filter_top_significant_points_ui(st.session_state.df_rm_anova, "rm_anova_plot")
            _df_rm_full = st.session_state.df_rm_anova
            if _df_rm_full is not None and not _df_rm_full.empty and "significant" in _df_rm_full.columns:
                _n_sig = int(_df_rm_full["significant"].sum())
                _total = len(_df_rm_full)
                st.write(f"Significant: {_n_sig}")
                st.write(f"Insignificant: {_total - _n_sig}")
                st.write(f"Total data points: {_total}")
            fig = get_rm_anova_plot(
                rm_anova_plot_df,
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
            _df_rm = st.session_state.df_rm_anova
            _p_col = next((c for c in ["p-corrected", "p"] if c in _df_rm.columns), None)
            if _p_col:
                all_metabolites = list(_df_rm.sort_values(_p_col)["metabolite"])
            else:
                all_metabolites = sorted(list(_df_rm["metabolite"]))

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

            st.markdown("---")
            st.markdown("#### Download Boxplots as PDF")
            st.caption("Exports 4 boxplots per page (2 × 2 grid) sorted by corrected p-value.")

            def _rma_clear_pdf_on_mode_change():
                st.session_state.pop("rma_boxplot_pdf_bytes", None)
                st.session_state.pop("rma_boxplot_pdf_label", None)

            _rma_pdf_mode = st.radio(
                "Selection mode",
                options=["Top N significant", "Top N insignificant", "Single metabolite"],
                horizontal=True,
                key="rma_boxplot_pdf_mode",
                on_change=_rma_clear_pdf_on_mode_change,
            )

            if _rma_pdf_mode in ("Top N significant", "Top N insignificant"):
                _rma_df_check = st.session_state.df_rm_anova
                _rma_want_sig = (_rma_pdf_mode == "Top N significant")
                if "significant" in _rma_df_check.columns:
                    _rma_avail = int(_rma_df_check["significant"].eq(_rma_want_sig).sum())
                else:
                    _rma_avail = len(_rma_df_check)
                _rma_max_n = max(5, min(20, _rma_avail))
                _rma_min_n = min(5, _rma_avail)
                if _rma_avail == 0:
                    st.warning(f"No {'significant' if _rma_want_sig else 'insignificant'} metabolites available.")
                    _rma_top_n = 0
                else:
                    _rma_top_n = st.slider(
                        "Number of metabolites to include",
                        min_value=_rma_min_n,
                        max_value=_rma_max_n,
                        value=min(10, _rma_max_n),
                        step=1,
                        key="rma_boxplot_pdf_topn",
                    )
                    if _rma_avail < 20:
                        st.caption(f"{_rma_avail} {'significant' if _rma_want_sig else 'insignificant'} metabolites available.")
            else:
                _rma_top_n = 1

            if st.button("Generate PDF", key="rma_generate_pdf_btn", type="primary"):
                _rma_df = st.session_state.df_rm_anova
                _rma_p_col = next((c for c in ["p-corrected", "p"] if c in _rma_df.columns), None)
                if _rma_pdf_mode == "Single metabolite":
                    _rma_mets = [st.session_state.rm_anova_metabolite]
                    _rma_label = f"single_{st.session_state.rm_anova_metabolite}"
                else:
                    if "significant" not in _rma_df.columns:
                        st.warning("Significance column not found in results.")
                        _rma_mets = []
                        _rma_label = ""
                    else:
                        _rma_pool = _rma_df[_rma_df["significant"] == _rma_want_sig]
                        if _rma_p_col:
                            _rma_pool = _rma_pool.sort_values(_rma_p_col)
                        _rma_mets = list(_rma_pool.index[:_rma_top_n])
                        _rma_label = f"top{_rma_top_n}_{'significant' if _rma_want_sig else 'insignificant'}"
                if _rma_mets:
                    with st.spinner(f"Generating PDF — {len(_rma_mets)} boxplot(s)…"):
                        _rma_pdf = generate_boxplot_pdf_generic(_rma_df, _rma_mets, get_rm_anova_metabolite_boxplot)
                    st.session_state["rma_boxplot_pdf_bytes"] = _rma_pdf
                    st.session_state["rma_boxplot_pdf_label"] = _rma_label
                else:
                    st.warning("No metabolites to include in the PDF.")
                    st.session_state.pop("rma_boxplot_pdf_bytes", None)

            if st.session_state.get("rma_boxplot_pdf_bytes"):
                st.download_button(
                    label="⬇ Download PDF",
                    data=st.session_state["rma_boxplot_pdf_bytes"],
                    file_name=f"rm_anova_{st.session_state.get('rma_boxplot_pdf_label', 'boxplots')}.pdf",
                    mime="application/pdf",
                    key="rma_download_pdf_btn",
                )

else:
    st.warning("⚠️ Please complete data preparation step first!")
