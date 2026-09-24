import streamlit as st
import pandas as pd
from src.common import *
from src.wilcoxon import *

page_setup()
st.session_state["current_page"] = "Wilcoxon Signed-Rank"

st.markdown("# Wilcoxon Signed-Rank Test")

with st.expander("📖 About"):
    st.markdown("""
The Wilcoxon signed-rank test is a non-parametric test for comparing **two paired or matched samples**. It is the direct non-parametric equivalent of the **paired t-test**, used when the normality assumption cannot be met.

##### 🧪 When to use it
- When your two groups consist of **paired or matched samples** (e.g., before vs. after treatment, same subject measured twice, matched case-control pairs).
- When your data are **not normally distributed** or contain outliers.
- As a robust alternative to the paired t-test.

> ⚠️ **Important:** This test requires a **subject/pairing column** in your metadata (e.g., patient ID, mouse ID) with the same value for the two matched samples. Samples are paired by this column; subjects that are not measured in both groups are excluded.

##### 🧪 Choosing between tests
| Data structure | Parametric | Non-parametric |
|---|---|---|
| Two independent groups | T-test (Student / Welch) | Mann-Whitney U |
| Two paired groups | Paired T-test | **Wilcoxon Signed-Rank** |
| Three or more groups | One-way ANOVA | Kruskal-Wallis |

##### ⚙️ Alternative hypotheses
- **Two-sided (default):** tests for a difference in *either* direction.
- **Greater:** tests if group A is systematically greater than group B.
- **Less:** tests if group A is systematically less than group B.

##### 📊 Key outputs
- **W_val** – Wilcoxon W statistic (for a two-sided test, the smaller of the positive and negative signed-rank sums).
- **p_val** – probability of observing a difference at least this extreme if there were truly no difference between the paired groups (p < 0.05 = significant).
- **RBC** – matched-pairs rank-biserial correlation; non-parametric effect size ranging from −1 to +1 (|0.1| = small, |0.3| = medium, |0.5| = large).
- **CLES** – common language effect size; probability that a randomly chosen observation from group A is greater than one from group B.
- **n pairs** – number of complete subject pairs used for the test.
- **p-corrected** – p-values adjusted for multiple comparisons across all tested metabolites, using the correction method selected in the sidebar.
- **Significance** – whether the corrected p-value is below 0.05.
    """)

# Initialise session state for results
if "df_wilcoxon" not in st.session_state:
    st.session_state.df_wilcoxon = pd.DataFrame()

if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)

    def clear_wilcoxon_data():
        st.session_state.df_wilcoxon = pd.DataFrame()
        st.session_state.pop("wilcoxon_attempted_metabolites", None)
        st.session_state.pop("wilcoxon_returned_metabolites", None)
        for key in list(st.session_state.keys()):
            if key.startswith("wilcoxon_metabolite") or key.startswith("_page_tab_"):
                del st.session_state[key]

    c1.selectbox(
        "select attribute of interest",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="wilcoxon_attribute",
        on_change=clear_wilcoxon_data,
    )

    if st.session_state.wilcoxon_attribute is not None:
        attribute_options = list(
            set(st.session_state.md[st.session_state.wilcoxon_attribute].dropna())
        )
        attribute_options.sort()
    else:
        attribute_options = []

    c2.multiselect(
        "select **two** options from the attribute for comparison",
        options=attribute_options,
        default=attribute_options[:2],
        key="wilcoxon_options",
        max_selections=2,
        help="Select two options.",
        on_change=clear_wilcoxon_data,
    )

    c1, c2 = st.columns(2)
    subject_options = [c for c in st.session_state.md.columns if c != st.session_state.wilcoxon_attribute]
    c1.selectbox(
        "subject / pairing column",
        options=subject_options,
        key="wilcoxon_subject",
        help="Metadata column that identifies which samples belong together (e.g., the same patient before and after treatment).",
        on_change=clear_wilcoxon_data,
    )
    c2.selectbox(
        "alternative",
        options=["two-sided", "greater", "less"],
        key="wilcoxon_alternative",
        help=(
            "Choose the test direction: 'two-sided' checks for any difference; "
            "'greater' tests if group A > group B; 'less' tests if group A < group B."
        ),
        on_change=clear_wilcoxon_data,
    )

    pairing_ok = False
    if len(st.session_state.wilcoxon_options) == 2 and st.session_state.get("wilcoxon_subject"):
        from src.utils import check_pairing
        n_pairs, duplicated_subjects = check_pairing(
            st.session_state.md, st.session_state.wilcoxon_attribute,
            st.session_state.wilcoxon_subject, st.session_state.wilcoxon_options,
        )
        if duplicated_subjects:
            st.error(
                f"Some subjects have more than one sample in the same group, so pairing is ambiguous: "
                f"{', '.join(duplicated_subjects[:10])}{' …' if len(duplicated_subjects) > 10 else ''}. "
                "Please choose a column that uniquely identifies each pair."
            )
        elif n_pairs < 2:
            st.error("Fewer than 2 subjects are measured in both groups. Please check the subject/pairing column.")
        else:
            st.info(f"{n_pairs} complete subject pairs found.")
            pairing_ok = True

    c1, _ = st.columns(2)
    run_disabled = len(st.session_state.wilcoxon_options) != 2 or not pairing_ok
    if c1.button("Run Wilcoxon Signed-Rank test", type="primary", disabled=run_disabled):
        progress_placeholder = st.empty()
        time_placeholder = st.empty()

        def progress_callback(done, total, est_left):
            progress_placeholder.progress(
                done / total, text=f"Running Wilcoxon Signed-Rank: metabolite {done} of {total}"
            )
            time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")

        result_df = gen_wilcoxon_data(
            st.session_state.wilcoxon_attribute,
            st.session_state.wilcoxon_options,
            st.session_state.wilcoxon_alternative,
            corrections_map[st.session_state.p_value_correction],
            st.session_state.wilcoxon_subject,
            _progress_callback=progress_callback,
        )
        progress_placeholder.empty()
        time_placeholder.empty()

        if result_df.empty:
            st.error(
                "No results were returned. Please ensure at least 2 subjects are measured in both groups "
                "and that the paired differences are not all zero."
            )
        else:
            st.session_state.df_wilcoxon = result_df
            st.rerun()

    wilcoxon_attempted = st.session_state.get("wilcoxon_attempted_metabolites")
    wilcoxon_returned = st.session_state.get("wilcoxon_returned_metabolites")
    if wilcoxon_attempted is not None and wilcoxon_returned is not None and wilcoxon_attempted != wilcoxon_returned:
        st.warning(
            f"Wilcoxon attempted {wilcoxon_attempted} metabolites, but only {wilcoxon_returned} produced plottable results. "
            f"{wilcoxon_attempted - wilcoxon_returned} metabolite(s) were skipped because valid test outputs could not be computed for the selected filters."
        )

    df = st.session_state.df_wilcoxon
    if df is not None and not df.empty:
        tabs = st.tabs(["📈 Feature significance", "📊 Single metabolite plots", "📁 Data"])

        with tabs[0]:
            color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
            st.selectbox("Color significant points by", options=color_by_options, key="wilcoxon_color_by")
            _wilcoxon_color = st.session_state.get("wilcoxon_color_by", "Significance (default)")
            wilcoxon_plot_df = filter_top_significant_points_ui(df, "wilcoxon_plot")
            fig = plot_wilcoxon(wilcoxon_plot_df, color_by=None if _wilcoxon_color == "Significance (default)" else _wilcoxon_color)
            show_fig(fig, "wilcoxon-feature-significance")
            st.session_state["page_figs_wilcoxon_sig"] = fig

        with tabs[1]:
            metabolite_options = list(df.index)

            def metabolite_label(m):
                return m.split("&")[0] if "&" in m else m

            st.selectbox(
                "metabolite",
                metabolite_options,
                key="wilcoxon_metabolite",
                format_func=metabolite_label,
            )
            if st.session_state.wilcoxon_metabolite in st.session_state.data.columns:
                fig = wilcoxon_boxplot(df, st.session_state.wilcoxon_metabolite)
                if fig is not None:
                    show_fig(
                        fig,
                        f"wilcoxon-boxplot-{st.session_state.wilcoxon_metabolite}",
                        container_width=True,
                    )
                    st.session_state["page_figs_wilcoxon_boxplot"] = fig
            else:
                st.warning(
                    "Selected metabolite not found in data columns. Please select a valid metabolite."
                )

            st.markdown("---")
            st.markdown("#### Download Boxplots as PDF")
            st.caption("Exports 4 boxplots per page (2 × 2 grid) sorted by corrected p-value.")

            def _wil_clear_pdf_on_mode_change():
                st.session_state.pop("wil_boxplot_pdf_bytes", None)
                st.session_state.pop("wil_boxplot_pdf_label", None)

            _wil_pdf_mode = st.radio(
                "Selection mode",
                options=["Top N significant", "Top N insignificant", "Single metabolite"],
                horizontal=True,
                key="wil_boxplot_pdf_mode",
                on_change=_wil_clear_pdf_on_mode_change,
            )

            if _wil_pdf_mode in ("Top N significant", "Top N insignificant"):
                _wil_want_sig = (_wil_pdf_mode == "Top N significant")
                _wil_sig_col = next((c for c in ["significance", "significant"] if c in df.columns), None)
                if _wil_sig_col:
                    _wil_avail = int(df[_wil_sig_col].eq(_wil_want_sig).sum())
                else:
                    _wil_avail = len(df)
                _wil_max_n = max(5, min(20, _wil_avail))
                _wil_min_n = min(5, _wil_avail)
                if _wil_avail == 0:
                    st.warning(f"No {'significant' if _wil_want_sig else 'insignificant'} metabolites available.")
                    _wil_top_n = 0
                else:
                    _wil_top_n = st.slider(
                        "Number of metabolites to include",
                        min_value=_wil_min_n,
                        max_value=_wil_max_n,
                        value=min(10, _wil_max_n),
                        step=1,
                        key="wil_boxplot_pdf_topn",
                    )
                    if _wil_avail < 20:
                        st.caption(f"{_wil_avail} {'significant' if _wil_want_sig else 'insignificant'} metabolites available.")
            else:
                _wil_top_n = 1

            if st.button("Generate PDF", key="wil_generate_pdf_btn", type="primary"):
                _wil_p_col = next((c for c in ["p-corrected", "p_val"] if c in df.columns), None)
                _wil_sig_col = next((c for c in ["significance", "significant"] if c in df.columns), None)
                if _wil_pdf_mode == "Single metabolite":
                    _wil_mets = [st.session_state.wilcoxon_metabolite]
                    _wil_label = f"single_{st.session_state.wilcoxon_metabolite}"
                else:
                    if not _wil_sig_col:
                        st.warning("Significance column not found in results.")
                        _wil_mets = []
                        _wil_label = ""
                    else:
                        _wil_pool = df[df[_wil_sig_col] == _wil_want_sig]
                        if _wil_p_col:
                            _wil_pool = _wil_pool.sort_values(_wil_p_col)
                        _wil_mets = list(_wil_pool.index[:_wil_top_n])
                        _wil_label = f"top{_wil_top_n}_{'significant' if _wil_want_sig else 'insignificant'}"
                if _wil_mets:
                    with st.spinner(f"Generating PDF — {len(_wil_mets)} boxplot(s)…"):
                        _wil_pdf = generate_boxplot_pdf_generic(df, _wil_mets, wilcoxon_boxplot)
                    st.session_state["wil_boxplot_pdf_bytes"] = _wil_pdf
                    st.session_state["wil_boxplot_pdf_label"] = _wil_label
                else:
                    st.warning("No metabolites to include in the PDF.")
                    st.session_state.pop("wil_boxplot_pdf_bytes", None)

            if st.session_state.get("wil_boxplot_pdf_bytes"):
                st.download_button(
                    label="⬇ Download PDF",
                    data=st.session_state["wil_boxplot_pdf_bytes"],
                    file_name=f"wilcoxon_{st.session_state.get('wil_boxplot_pdf_label', 'boxplots')}.pdf",
                    mime="application/pdf",
                    key="wil_download_pdf_btn",
                )

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
            for col in ["p_val", "p-corrected"]:
                if col in df_display.columns:
                    style_dict[col] = sci_notation_or_plain
            if style_dict:
                styled = df_display.style.format(style_dict)
                st.dataframe(styled, use_container_width=True)
            else:
                st.dataframe(df_display, use_container_width=True)

else:
    st.warning("⚠️ Please complete data preparation step first!")
