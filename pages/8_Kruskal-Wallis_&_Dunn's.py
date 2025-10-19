import streamlit as st
from src.common import *
from src.kruskal import *

page_setup()

st.markdown("# Kruskal Wallis & Dunn's post hoc")

with st.expander("📖 About"):
    st.markdown(
        """The Kruskal-Wallis test helps determine if there are significant differences among multiple groups, and if significant differences exist, Dunn's post hoc test helps pinpoint which specific groups differ from each other. These non-parametric tests are valuable tools for analyzing data when the assumptions of parametric tests are not met or when working with ordinal or skewed data."""
    )
    st.image("assets/figures/kruskal-wallis.png")
    st.image("assets/figures/dunn.png")

if not st.session_state.data.empty:
    c1, c2 = st.columns(2)

    prev_kruskal_attribute = st.session_state.get("_prev_kruskal_attribute", None)
    kruskal_attribute = c1.selectbox(
        "attribute for Kruskal Wallis test",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="kruskal_attribute",
    )

    if prev_kruskal_attribute is not None and kruskal_attribute != prev_kruskal_attribute:
        st.session_state.df_kruskal = pd.DataFrame()
        st.session_state.df_dunn = pd.DataFrame()
    
    st.session_state["_prev_kruskal_attribute"] = kruskal_attribute

    attribute = st.session_state.kruskal_attribute
    attribute_options = list(set(st.session_state.md[attribute].dropna()))
    attribute_options.sort()

    prev_kruskal_groups = st.session_state.get("_prev_kruskal_groups", None)
    kruskal_groups = c1.multiselect(
        "Select groups to include in Kruskal Wallis (minimum 3)",
        options=attribute_options,
        default=attribute_options,
        key="kruskal_groups",
        help="For comparing 2 groups, use the t-test page instead",
    )

    if prev_kruskal_groups is not None and set(kruskal_groups) != set(prev_kruskal_groups):
        st.session_state.df_kruskal = pd.DataFrame()
        st.session_state.df_dunn = pd.DataFrame()
    st.session_state["_prev_kruskal_groups"] = list(kruskal_groups)

    min_required = 3
    run_disabled = not ("kruskal_groups" in st.session_state and len(st.session_state.kruskal_groups) >= min_required)

    c1.button("Run Kruskal Wallis", key="run_kruskal", type="primary", disabled = run_disabled)
    if st.session_state.run_kruskal:
        if "kruskal_groups" not in st.session_state or len(st.session_state.kruskal_groups) < min_required:
            st.error(f"At least {min_required} groups must be selected to run Kruskal Wallis.")
        else:
            st.session_state.df_kruskal = kruskal_wallis(
                st.session_state.data,
                st.session_state.kruskal_attribute,
                corrections_map[st.session_state.p_value_correction],
                elements=st.session_state.kruskal_groups
            )
            st.rerun()

    if not st.session_state.df_kruskal.empty:
        dunn_options = list(st.session_state.kruskal_groups) if "kruskal_groups" in st.session_state else []
        dunn_options.sort()

        prev_kruskal_elements = st.session_state.get("_prev_dunns_options", None)
        dunn_elements = c2.multiselect(
            "select **two** options for Dunn's comparison",
            options=dunn_options,
            default=dunn_options[:2],
            key="dunn_elements",
            max_selections=2,
            help="Select two options.",
        )

        if prev_kruskal_elements is not None and set(dunn_elements) != set(prev_kruskal_elements):
            st.session_state.df_dunn = pd.DataFrame()
        st.session_state["_prev_dunns_options"] = list(dunn_elements)
        c2.button(
            "Run Dunn's",
            key="run_dunn",
            type="primary",
            disabled=len(st.session_state.dunn_elements) != 2,
        )

        if st.session_state.run_dunn:
            st.session_state.df_dunn = dunn(
                st.session_state.df_kruskal,
                st.session_state.kruskal_attribute,
                st.session_state.dunn_elements,
                corrections_map[st.session_state.p_value_correction]
            )
            st.rerun()

    tab_options = [
        "📈 KW: plot",
        "📁 KW: result table",
        "📊 KW: metabolites (boxplots)",
    ]

    if not st.session_state.df_dunn.empty:
        tab_options += ["📈 Dunn's: plot", "📁 Dunn's: result table"]


    if not st.session_state.df_kruskal.empty:
        tabs = st.tabs(tab_options)
        with tabs[0]:
            fig = get_kruskal_plot(st.session_state.df_kruskal)
            show_fig(fig, "kruskal")
        with tabs[1]:
            df_display = st.session_state.df_kruskal.copy()
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
            #index column is metabolite name
            for col in ["p", "p-corrected"]:
                if col in df_display.columns:
                    style_dict[col] = sci_notation_or_plain
            if style_dict:
                styled = df_display.style.format(style_dict)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        with tabs[2]:
            # Include both significant and insignificant metabolites in dropdown
            all_metabolites = sorted(list(st.session_state.df_kruskal["metabolite"]))
            st.selectbox(
                "select metabolite",
                all_metabolites,
                key="kruskal_metabolite",
            )

            met = st.session_state.kruskal_metabolite
            df_kruskal = st.session_state.df_kruskal
            if met in df_kruskal.index and "significant" in df_kruskal.columns:
                is_sig = df_kruskal.loc[met, "significant"]
                desc = "Significant" if is_sig else "Insignificant"
                st.write(f"**{desc} Metabolite: {met}**")
            
            fig = get_metabolite_boxplot(
                st.session_state.df_kruskal,
                st.session_state.kruskal_metabolite,
            )
            
            show_fig(fig, f"kruskal-{st.session_state.kruskal_metabolite}")
        
        if not st.session_state.df_dunn.empty:
            with tabs[3]:
                #st.write("Dunn's test plots are not yet implemented.")
                fig1 = get_dunn_teststat_plot(getattr(st.session_state.df_dunn, '_original', st.session_state.df_dunn))
                show_fig(fig1, "dunn-teststat")
                fig2 = get_dunn_volcano_plot(getattr(st.session_state.df_dunn, '_original', st.session_state.df_dunn))
                show_fig(fig2, "dunn-volcano")
            with tabs[4]:
                df_dunn = st.session_state.df_dunn.copy()
                df_dunn = df_dunn.rename(columns={
                    "stats_metabolite": "metabolite",
                    "stats_significant": "significant"
                })

                attribute_value = st.session_state.kruskal_attribute if "kruskal_attribute" in st.session_state else ""
                dunn_groups = st.session_state.dunn_elements if "dunn_elements" in st.session_state else ["", ""]
                if len(dunn_groups) == 2:
                    A_value, B_value = dunn_groups
                else:
                    A_value, B_value = "", ""
                sig_idx = df_dunn.columns.get_loc("significant") + 1 if "significant" in df_dunn.columns else len(df_dunn.columns)
                df_dunn.insert(sig_idx, "attribute", attribute_value)
                df_dunn.insert(sig_idx + 1, "A", A_value)
                df_dunn.insert(sig_idx + 2, "B", B_value)

                def sci_notation_or_plain(x):
                    try:
                        if pd.isnull(x):
                            return x
                        if isinstance(x, str):
                            return x
                        if float(x) == 0:
                            return 0
                        return f"{x:.2e}"
                    except Exception:
                        return x
                style_dict = {}
                for col in ["p", "p-corrected"]:
                    if col in df_dunn.columns:
                        style_dict[col] = sci_notation_or_plain
                if style_dict:
                    styled = df_dunn.style.format(style_dict)
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_dunn, use_container_width=True, hide_index=True)
else:
    st.warning(
        "Please complete data clean up step first! (Preparing data for statistical analysis)"
    )
