import streamlit as st
import pandas as pd

from src.common import *
from src.ttest import *

page_setup()
st.session_state["current_page"] = "T-test"

st.markdown("# T-test")

with st.expander("📖 About"):
    st.markdown("""
This module compares the means of two groups to assess whether they differ significantly.

##### 🧪 Choosing the right test
- **Student's t-test** – the classic and most widely used version; assumes both groups have *equal variances* and are *normally distributed*. It's suitable for balanced datasets.  
- **Welch's t-test** – a more robust variant that does *not* assume equal variances or sample sizes. It's recommended for most real-world biological data.  
- **Paired t-test** – used when both measurements come from the **same or matched samples** (e.g., before vs. after treatment).  
- **Auto** – defaults to **Welch's t-test**. You can check the **Parametric Assumptions Evaluation** page to confirm whether equal variances hold in your data.  

##### ⚙️ Alternative hypotheses
- **Two-sided (default):** tests whether the two means differ in *either direction*.  
- **Greater:** tests if group A > group B.  
- **Less:** tests if group A < group B.  
Most studies use **two-sided** unless there's a strong directional expectation.

##### 📊 Key outputs
- **T** – test statistic measuring difference magnitude relative to variability.  
- **p-val** – probability that the observed difference is due to chance (p < 0.05 = significant).  
- **dof** – degrees of freedom, based on group sizes and test type.  
- **Cohen's d** – effect size (0.2 = small, 0.5 = medium, 0.8 = large).  
- **BF10** – Bayes Factor showing evidence for the alternative hypothesis (> 3 = moderate evidence).  
- **Power** – likelihood of correctly detecting a true difference.  
- **p-corrected (FDR)** – adjusted p-values accounting for multiple comparisons across all metabolites.  
- **Significance** – marks whether the adjusted result remains significant (after FDR).  
- **ttest_type** – identifies which test (Student, Welch, or Paired) was applied.

##### Why apply FDR correction?
Even though the t-test compares only two groups, each metabolite is tested separately — often hundreds or thousands at once.  
This creates a *multiple-testing problem*, where some features appear significant by chance.  
The **False Discovery Rate (FDR)** correction adjusts for this, helping ensure that identified metabolites remain statistically significant after accounting for multiple comparisons.
    """)

# Ensure st.session_state.df_ttest is initialized
if "df_ttest" not in st.session_state:
    st.session_state.df_ttest = pd.DataFrame()

if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)
    def clear_ttest_data():
        st.session_state.df_ttest = pd.DataFrame()
        st.session_state.pop("ttest_attempted_metabolites", None)
        st.session_state.pop("ttest_returned_metabolites", None)
        # Remove tab-related session state if present
        for key in list(st.session_state.keys()):
            if key.startswith("ttest_metabolite") or key.startswith("_page_tab_"):
                del st.session_state[key]

    c1.selectbox(
        "select attribute of interest",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="ttest_attribute",
        on_change=clear_ttest_data
    )

    if st.session_state.ttest_attribute is not None:
        attribute_options = list(
            set(st.session_state.md[st.session_state.ttest_attribute].dropna())
        )
        attribute_options.sort()
    else:
        attribute_options = []
    c2.multiselect(
        "select **two** options from the attribute for comparison",
        options=attribute_options,
        default=attribute_options[:2],
        key="ttest_options",
        max_selections=2,
        help="Select two options.",
        on_change=clear_ttest_data
    )
    c1, c2, c3 = st.columns(3)
    v_space(2, c1)
    c1.checkbox(
        "paired",
        False,
        key="ttest_paired",
        help="Specify whether the two observations are related (i.e. repeated measures) or independent.",
        on_change=clear_ttest_data
    )

    correction_options = {
        "auto": "auto",
        "Welch's": "True",
        "Student's": "False"
    }
    c2.selectbox(
        "T-test type",
        options=list(correction_options.keys()),
        key="ttest_correction_label",
        help="Welch's (recommended) ignores equal-variance assumption. Use Student's if variances are equal. 'Auto' applies Welch by default; check Levene test for confirmation.",
        on_change=clear_ttest_data
    )

    c3.selectbox(
        "alternative",
        options=["two-sided", "greater", "less"],
        key="ttest_alternative",
        help="Choose the test direction: 'two-sided' checks for any difference; 'greater' tests if group A > group B; and 'less' tests if group A < group B.",
        on_change=clear_ttest_data
    )

    if c1.button("Run t-test", type="primary", disabled=(len(st.session_state.ttest_options) != 2)):
        # Map label to value for correction
        correction_value = correction_options[st.session_state.ttest_correction_label]
        
        # Add progress bar
        progress_placeholder = st.empty()
        time_placeholder = st.empty()
        
        def progress_callback(done, total, est_left):
            progress = done / total
            progress_placeholder.progress(progress, text=f"Running t-test: metabolite {done} of {total}")
            time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")

        st.session_state.df_ttest = gen_ttest_data(
            st.session_state.ttest_attribute,
            st.session_state.ttest_options,
            st.session_state.ttest_paired,
            st.session_state.ttest_alternative,
            correction_value,
            corrections_map[st.session_state.p_value_correction],
            _progress_callback=progress_callback
        )
        
        progress_placeholder.empty()
        time_placeholder.empty()
        st.rerun()

    ttest_attempted = st.session_state.get("ttest_attempted_metabolites")
    ttest_returned = st.session_state.get("ttest_returned_metabolites")
    if ttest_attempted is not None and ttest_returned is not None and ttest_attempted != ttest_returned:
        st.warning(
            f"T-test attempted {ttest_attempted} metabolites, but only {ttest_returned} produced plottable results. "
            f"{ttest_attempted - ttest_returned} metabolite(s) were skipped because valid test outputs could not be computed for the selected filters."
        )

    # Only show tabs if t-test results exist (button pressed and results generated)
    ttest_stat_cols = {"T", "T-val", "t", "tval"}
    df = st.session_state.df_ttest
    if df is not None and not df.empty:
        tabs = st.tabs(["📈 Feature significance", "📈 Volcano plot", "📊 Single metabolite plots", "📁 Data"])

        with tabs[0]:
            color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
            st.selectbox("Color significant points by", options=color_by_options, key="ttest_color_by")
            _ttest_color = st.session_state.get("ttest_color_by", "Significance (default)")
            ttest_plot_df = filter_top_significant_points_ui(df, "ttest_plot")
            fig = plot_ttest(ttest_plot_df, color_by=None if _ttest_color == "Significance (default)" else _ttest_color)
            show_fig(fig, "t-test")
            st.session_state["page_figs_ttest_sig"] = fig

        with tabs[1]:
            # Volcano Plot tab 
            if "mean(A)" in df.columns and "mean(B)" in df.columns:
                fig_volcano = get_ttest_volcano_plot(df)
                show_fig(fig_volcano, "ttest-volcano")
                st.session_state["page_figs_ttest_volcano"] = fig_volcano
            else:
                st.warning("Could not generate volcano plot. Mean values are missing from t-test results. Please re-run the t-test.")


        with tabs[2]:
            metabolite_options = list(df.index)
            def metabolite_label(m):
                return m.split("&")[0] if "&" in m else m
            
            st.selectbox("metabolite", metabolite_options, key="ttest_metabolite", format_func=metabolite_label)
            # Only plot if the selected metabolite is in the data columns
            if st.session_state.ttest_metabolite in st.session_state.data.columns:
                fig = ttest_boxplot(df, st.session_state.ttest_metabolite)
                if fig is not None:
                    show_fig(fig, f"ttest-boxplot-{st.session_state.ttest_metabolite}", container_width=True)
                    st.session_state["page_figs_ttest_boxplot"] = fig
            else:
                st.warning(f"Selected metabolite not found in data columns. Please select a valid metabolite.")

            st.markdown("---")
            st.markdown("#### Download Boxplots as PDF")
            st.caption("Exports 4 boxplots per page (2 × 2 grid) sorted by corrected p-value.")

            def _ttest_clear_pdf_on_mode_change():
                st.session_state.pop("ttest_boxplot_pdf_bytes", None)
                st.session_state.pop("ttest_boxplot_pdf_label", None)

            _ttest_pdf_mode = st.radio(
                "Selection mode",
                options=["Top N significant", "Top N insignificant", "Single metabolite"],
                horizontal=True,
                key="ttest_boxplot_pdf_mode",
                on_change=_ttest_clear_pdf_on_mode_change,
            )

            if _ttest_pdf_mode in ("Top N significant", "Top N insignificant"):
                _ttest_want_sig = (_ttest_pdf_mode == "Top N significant")
                _sig_col = next((c for c in ["significance", "significant"] if c in df.columns), None)
                if _sig_col:
                    _ttest_avail = int(df[_sig_col].eq(_ttest_want_sig).sum())
                else:
                    _ttest_avail = len(df)
                _ttest_max_n = max(5, min(20, _ttest_avail))
                _ttest_min_n = min(5, _ttest_avail)
                if _ttest_avail == 0:
                    st.warning(f"No {'significant' if _ttest_want_sig else 'insignificant'} metabolites available.")
                    _ttest_top_n = 0
                else:
                    _ttest_top_n = st.slider(
                        "Number of metabolites to include",
                        min_value=_ttest_min_n,
                        max_value=_ttest_max_n,
                        value=min(10, _ttest_max_n),
                        step=1,
                        key="ttest_boxplot_pdf_topn",
                    )
                    if _ttest_avail < 20:
                        st.caption(f"{_ttest_avail} {'significant' if _ttest_want_sig else 'insignificant'} metabolites available.")
            else:
                _ttest_top_n = 1

            if st.button("Generate PDF", key="ttest_generate_pdf_btn", type="primary"):
                _ttest_p_col = next((c for c in ["p-corrected", "p-val"] if c in df.columns), None)
                _sig_col = next((c for c in ["significance", "significant"] if c in df.columns), None)
                if _ttest_pdf_mode == "Single metabolite":
                    _ttest_mets = [st.session_state.ttest_metabolite]
                    _ttest_label = f"single_{st.session_state.ttest_metabolite}"
                else:
                    if not _sig_col:
                        st.warning("Significance column not found in results.")
                        _ttest_mets = []
                        _ttest_label = ""
                    else:
                        _ttest_pool = df[df[_sig_col] == _ttest_want_sig]
                        if _ttest_p_col:
                            _ttest_pool = _ttest_pool.sort_values(_ttest_p_col)
                        _ttest_mets = list(_ttest_pool.index[:_ttest_top_n])
                        _ttest_label = f"top{_ttest_top_n}_{'significant' if _ttest_want_sig else 'insignificant'}"
                if _ttest_mets:
                    with st.spinner(f"Generating PDF — {len(_ttest_mets)} boxplot(s)…"):
                        _ttest_pdf = generate_boxplot_pdf_generic(df, _ttest_mets, ttest_boxplot)
                    st.session_state["ttest_boxplot_pdf_bytes"] = _ttest_pdf
                    st.session_state["ttest_boxplot_pdf_label"] = _ttest_label
                else:
                    st.warning("No metabolites to include in the PDF.")
                    st.session_state.pop("ttest_boxplot_pdf_bytes", None)

            if st.session_state.get("ttest_boxplot_pdf_bytes"):
                st.download_button(
                    label="⬇ Download PDF",
                    data=st.session_state["ttest_boxplot_pdf_bytes"],
                    file_name=f"ttest_{st.session_state.get('ttest_boxplot_pdf_label', 'boxplots')}.pdf",
                    mime="application/pdf",
                    key="ttest_download_pdf_btn",
                )


        with tabs[3]:
            # Format p-val and p-corrected columns in scientific notation, but allow full value on cell click
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