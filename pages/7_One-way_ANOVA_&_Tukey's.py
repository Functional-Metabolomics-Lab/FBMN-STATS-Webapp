from src.common import *
from src.anova import *


page_setup()

st.markdown("# One-way ANOVA & Tukey's")

with st.expander("📖 About"):
    st.markdown(
        """Analysis of variance (ANOVA) is a statistical method used to compare means between two or more groups. ANOVA tests whether there is a significant difference between the means of different groups based on the variation within and between groups. If ANOVA reveals that there is a significant difference between at least two group means, post hoc tests are used to determine which specific groups differ significantly from one another. Tukey's post hoc test is a widely used statistical method for pairwise comparisons after ANOVA. It accounts for multiple comparisons and adjusts the p-values accordingly, allowing for a more accurate identification of significant group differences."""
    )
    st.image("assets/figures/anova.png")
    st.image("assets/figures/tukeys.png")

if not st.session_state.data.empty:
    c1, c2 = st.columns(2)

    prev_anova_attribute = st.session_state.get("_prev_anova_attribute", None)
    anova_attribute = c1.selectbox(
        "attribute for ANOVA test",
        options=[c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1],
        key="anova_attribute",
    )

    if prev_anova_attribute is not None and anova_attribute != prev_anova_attribute:
        st.session_state.df_anova = pd.DataFrame()
        st.session_state.df_tukey = pd.DataFrame()
    st.session_state["_prev_anova_attribute"] = anova_attribute

    attribute = st.session_state.anova_attribute
    attribute_options = list(set(st.session_state.md[attribute].dropna()))
    attribute_options.sort()

    prev_anova_groups = st.session_state.get("_prev_anova_groups", None)
    anova_groups = c1.multiselect(
        "Select groups to include in ANOVA (minimum 3)",
        options=attribute_options,
        default=attribute_options,
        key="anova_groups",
        help="For comparing 2 groups, use the t-test page instead",
    )

    if prev_anova_groups is not None and set(anova_groups) != set(prev_anova_groups):
        st.session_state.df_anova = pd.DataFrame()
        st.session_state.df_tukey = pd.DataFrame()
    st.session_state["_prev_anova_groups"] = list(anova_groups)

    min_required = 3
    run_disabled = not ("anova_groups" in st.session_state and len(st.session_state.anova_groups) >= min_required)

    c1.button("Run ANOVA", key="run_anova", type="primary", disabled=run_disabled)
    if st.session_state.run_anova:
        if "anova_groups" not in st.session_state or len(st.session_state.anova_groups) < min_required:
            st.error(f"At least {min_required} groups must be selected to run ANOVA.")
        else:
            st.session_state.df_anova = anova(
                st.session_state.data,
                st.session_state.anova_attribute,
                corrections_map[st.session_state.p_value_correction],
                elements=st.session_state.anova_groups,
            )
            st.rerun()

    if not st.session_state.df_anova.empty:
        tukey_options = list(st.session_state.anova_groups) if "anova_groups" in st.session_state else []
        tukey_options.sort()

        prev_tukey_elements = st.session_state.get("_prev_tukeys_options", None)
        tukey_elements = c2.multiselect(
            "select **two** options for Tukey's comparison",
            options=tukey_options,
            default=tukey_options[:2],
            key="tukeys_options",
            max_selections=2,
            help="Select two options.",
        )

        if prev_tukey_elements is not None and set(tukey_elements) != set(prev_tukey_elements):
            st.session_state.df_tukey = pd.DataFrame()
        st.session_state["_prev_tukeys_options"] = list(tukey_elements)
        st.session_state.tukey_elements = tukey_elements

        c2.button(
            "Run Tukey's",
            key="run_tukey",
            type="primary",
            disabled=len(st.session_state.tukey_elements) != 2,
        )
        # Only use existing ANOVA results, NOT rerun ANOVA calculations
        if st.session_state.run_tukey:
            st.session_state.df_tukey = tukey(
                st.session_state.df_anova,
                st.session_state.anova_attribute,
                st.session_state.tukey_elements,
                corrections_map[st.session_state.p_value_correction]
            )
            st.rerun()

    tab_options = [
        "📈 ANOVA: plot",
        "📁 ANOVA: result table",
        "📊 ANOVA: metabolites (boxplots)",
    ]

    if not st.session_state.df_tukey.empty:
        tab_options += ["📈 Tukey's: plots", "📁 Tukey's: result table"]

    if not st.session_state.df_anova.empty:
        tabs = st.tabs(tab_options)
        with tabs[0]:
            fig = get_anova_plot(st.session_state.df_anova)
            show_fig(fig, "anova")
        with tabs[1]:
            df_display = st.session_state.df_anova.copy()
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
                st.dataframe(styled, use_container_width=True)
            else:
                st.dataframe(df_display, use_container_width=True)
        with tabs[2]:
            ft = st.session_state.get("ft_gnps", pd.DataFrame())
            name_cols = [c for c in ("metabolite_name", "name", "feature_name", "compound_name", "compound") if c in ft.columns]
            if name_cols:
                name_col = name_cols[0]
                names = sorted(list(set(ft[name_col].dropna())))
                sel_name = st.selectbox("Filter by feature name (optional)", options=["-- All --"] + names, key="anova_name_filter")
                if sel_name and sel_name != "-- All --":
                    candidates = [idx for idx, row in ft[name_col].items() if row == sel_name and idx in st.session_state.df_anova.index]
                else:
                    candidates = list(st.session_state.df_anova.index)
            else:
                candidates = list(st.session_state.df_anova.index)

            candidates.sort()
            st.selectbox(
                "select metabolite",
                options=candidates,
                key="anova_metabolite",
            )

            met = st.session_state.anova_metabolite
            df_anova = st.session_state.df_anova
            if met in df_anova.index and "significant" in df_anova.columns:
                is_sig = df_anova.loc[met, "significant"]
                desc = "Significant" if is_sig else "Insignificant"
                st.write(f"**{desc} Metabolite: {met}**")

            fig = get_metabolite_boxplot(
                st.session_state.df_anova,
                st.session_state.anova_metabolite,
            )

            show_fig(fig, f"anova-{st.session_state.anova_metabolite}")

        if not st.session_state.df_tukey.empty:
            with tabs[3]:
                # Use original numeric values for plotting
                fig1 = get_tukey_teststat_plot(getattr(st.session_state.df_tukey, '_original', st.session_state.df_tukey))
                show_fig(fig1, "tukeys-teststat")
                fig2 = get_tukey_volcano_plot(getattr(st.session_state.df_tukey, '_original', st.session_state.df_tukey))
                show_fig(fig2, "tukeys-volcano")
            with tabs[4]:
                df_tukey = st.session_state.df_tukey.copy()
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
                    if col in df_tukey.columns:
                        style_dict[col] = sci_notation_or_plain
                if style_dict:
                    styled = df_tukey.style.format(style_dict)
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_tukey, use_container_width=True, hide_index=True)

else:
    st.warning(
        "Please complete data clean up step first! (Preparing data for statistical analysis)"
    )
