import streamlit as st

from src.common import *
from src.testparametric import *

page_setup("Parametric_assumptions_evaluation")

st.markdown("# Parametric Assumptions Evaluation")
st.markdown("## Normal Distribution and Equal Variance")

with st.expander("📖 Why is this important?"):
    st.markdown(
        """
        Before running statistical tests such as *t-tests* or *ANOVA*, it's important to verify whether your data meet the assumptions of **normal distribution** and **equal variances** between groups.  
        This page helps you assess these assumptions for your selected two groups.

        ##### 🧪 How to interpret the histograms?

        - The **x-axis** shows the *p-value range* (0 - 1).  
        - The **y-axis** shows the *number of features* (metabolites) that fall within each p-value range.  
        - Bars toward the **right (p > 0.05)** indicate features that likely satisfy the assumption.  
        - Bars toward the **left (p < 0.05)** indicate features that likely violate the assumption.

        **Normality (Shapiro–Wilk test)**  
        - Tests whether data for each feature are *normally distributed*.  
        - If most p-values are **> 0.05**, the data are approximately normal.  
        - If many are **< 0.05**, the data deviate from normality, so consider *non-parametric* tests.

        **Equal variance (Levene’s test)**  
        - Tests whether variances between groups are *equal*.  
        - If most p-values are **> 0.05**, the variances can be treated as equal.  
        - If many are **< 0.05**, variances differ → use *Welch’s t-test* or *non-parametric* methods.

        ##### 🧭 Choosing the right test based on results

        | Normality | Equal variance | Recommended test |
        |------------|----------------|------------------|
        | ✅ Normal | ✅ Equal | **Student's t-test** or **ANOVA** |
        | ✅ Normal | ❌ Unequal | **Welch's t-test** |
        | ❌ Non-normal | ✅ or ❌ | **Mann–Whitney U test** or **Kruskal–Wallis test** |

        💡 *Tip:*  
        If most bars are concentrated on the **right side (p > 0.05)** of both histograms, parametric tests like *Student's t-test* or *ANOVA* are suitable. If they cluster on the **left (p < 0.05)**, non-parametric tests such as *Kruskal–Wallis* or *Mann–Whitney U* are more appropriate.
""")


if st.session_state.data is not None and not st.session_state.data.empty:
    c1, c2 = st.columns(2)
    c1.selectbox(
        "select attribute of interest",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="test_attribute",
    )

    # Check if test_attribute is valid before accessing DataFrame
    if (
        st.session_state.test_attribute is None
        or st.session_state.test_attribute not in st.session_state.md.columns
    ):
        st.warning("Please select a valid attribute for parametric assumption evaluation.")
        st.stop()

    attribute_options = list(
        set(st.session_state.md[st.session_state.test_attribute].dropna())
    )
    attribute_options.sort()
    c2.multiselect(
        "select **two** options from the attribute for comparison",
        options=attribute_options,
        default=attribute_options[:2],
        key="test_options",
        max_selections=2,
        help="Select two options.",
    )
    if st.session_state.test_attribute and len(st.session_state.test_options) == 2:
        tabs = st.tabs(["📊 Normal distribution (Shapiro-Wilk test)", "📊 Equal variance (Levene test)"])
        with tabs[0]:
            fig = test_normal_distribution(st.session_state.test_attribute, st.session_state.test_options, corrections_map[st.session_state.p_value_correction])
            if fig:
                show_fig(fig, "test-normal-distribution")
        with tabs[1]:
            fig = test_equal_variance(st.session_state.test_attribute, st.session_state.test_options, corrections_map[st.session_state.p_value_correction])
            show_fig(fig, "test-equal-variance")

    with st.expander("📖 How to interpret the results?"):
        st.info(
                """💡 **Interpretation** In both tests low p-values indicate that data points for a feature are **NOT** normal distributed or have similar variance. To meet **parametric** criteria the p-values in the histograms should not be smaller than 0.05.When a larger number of data points indicate low p-values, it would be advisable to opt for a **non-parametric** statistical test. """ )
        st.image("assets/figures/decision.png")  
        
else:
    st.warning("⚠️ Please complete data preparation step first!")