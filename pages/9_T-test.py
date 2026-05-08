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