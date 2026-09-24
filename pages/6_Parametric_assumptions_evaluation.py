import streamlit as st

from src.common import *
from src.testparametric import *

page_setup()
st.session_state["current_page"] = "Parametric Assumptions Evaluation"

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

        **Equal variance (Bartlett’s test if data are normal, Levene’s test if not)**
        - Tests whether variances between groups are *equal*.
        - If most p-values are **> 0.05**, the variances can be treated as equal.
        - If many are **< 0.05**, variances differ → use *Welch’s t-test* or *non-parametric* methods.

        ##### 🧭 Choosing the right test based on results (protocol Fig. 12)

        | Data | Groups | Two groups | More than two groups |
        |------|--------|------------|----------------------|
        | ✅ Normal | independent | **t-test** (Welch's if variances differ) | **One-way ANOVA** → Tukey's post hoc |
        | ✅ Normal | paired / dependent | **Paired t-test** | **Repeated measures ANOVA** |
        | ❌ Not normal | independent | **Mann–Whitney U** | **Kruskal–Wallis** → Dunn's post hoc |
        | ❌ Not normal | paired / dependent | **Wilcoxon signed-rank** | **Friedman** |

        💡 *Tip:*  
        If most bars are concentrated on the **right side (p > 0.05)** of both histograms, parametric tests like *Student's t-test* or *ANOVA* are suitable. If they cluster on the **left (p < 0.05)**, non-parametric tests such as *Kruskal–Wallis* or *Mann–Whitney U* are more appropriate.
""")
    
with st.expander("📖 How to interpret the results?"):
    st.info(
            """💡 **Interpretation** In both tests, low p-values indicate that the data for a feature are **NOT** normally distributed (Shapiro-Wilk) or do **NOT** have equal variances (Levene/Bartlett). To meet **parametric** criteria the p-values in the histograms should not be smaller than 0.05. When a large number of features show low p-values, it is advisable to opt for a **non-parametric** statistical test. As in the protocol, the p-values are corrected for multiple testing across features with the method selected in the sidebar (Benjamini-Hochberg by default). Note that this page checks the two selected groups; for ANOVA, repeat the check for the other groups as well.""" )
    st.image("assets/figures/decision.png") 


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
        tab_normality, tab_variance = st.tabs(["📊 Normality Check", "📊 Equal Variance Check"])
        with tab_normality:
            fig = test_normal_distribution(st.session_state.data, st.session_state.md, st.session_state.test_attribute, st.session_state.test_options, corrections_map[st.session_state.p_value_correction])
            if fig:
                show_fig(fig, "test-normal-distribution")
                st.session_state["pae_normality_fig"] = fig

        with tab_variance:
            @st.fragment
            def equal_variance_section():
                variance_test = st.radio(
                    "Select equal variance test depending on Normality Check Results",
                    options=["Levene test", "Bartlett test"],
                    horizontal=True,
                    help="Levene's test is robust to non-normality. Bartlett's test is more powerful when data are normally distributed.",
                )
                if variance_test == "Levene test":
                    fig = test_equal_variance(st.session_state.data, st.session_state.md, st.session_state.test_attribute, st.session_state.test_options, corrections_map[st.session_state.p_value_correction])
                    show_fig(fig, "test-equal-variance")
                    st.session_state["pae_variance_fig"] = fig
                else:
                    fig = test_equal_variance_bartlett(st.session_state.data, st.session_state.md, st.session_state.test_attribute, st.session_state.test_options, corrections_map[st.session_state.p_value_correction])
                    show_fig(fig, "test-equal-variance-bartlett")
                    st.session_state["pae_variance_fig"] = fig

            equal_variance_section()
        
else:
    st.warning("⚠️ Please complete data preparation step first!")