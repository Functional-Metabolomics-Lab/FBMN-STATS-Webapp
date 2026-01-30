from src.common import *
from src.anova import *


page_setup("One-way_ANOVA_and_Tukeys_test")

st.markdown("# One-way ANOVA & Tukey's")

with st.expander("📖 About"):
    st.markdown(
        """
        ANOVA (Analysis of Variance) is used to test whether the means of multiple groups differ significantly. 
        It evaluates variation within and between groups to determine if **at least one group mean is different**.
        When ANOVA indicates a significant difference, a Tukey’s post-hoc test can be used to compare specific pairs of groups.
        In this app, users can select any two groups to perform Tukey’s test, which adjusts for multiple comparisons and identifies whether their means differ significantly.
       """
       )
    
    st.image("assets/figures/anova.png")
    st.image("assets/figures/tukeys.png")

if st.session_state.data is not None and not st.session_state.data.empty:

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
    # Check if attribute is valid before accessing DataFrame
    if attribute is None or attribute not in st.session_state.md.columns:
        st.warning("Please select a valid attribute for ANOVA. The attribute must be a column from your metadata file that contains at least two unique, non-missing group values. If you do not see your desired attribute, please check your metadata for missing or identical values.")
        st.stop()

    attribute_options = list(set(st.session_state.md[attribute].dropna()))
    attribute_options.sort()

    prev_anova_groups = st.session_state.get("_prev_anova_groups", None)
    anova_groups = c1.multiselect(
        "Select groups to include in ANOVA (minimum 3)",
        options=attribute_options,
        default=attribute_options,
        key="anova_groups",
        help="For comparing 2 groups, use the t-test page instead. If button is disabled, select a different attribute.",
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
            import time
            progress_placeholder = st.empty()
            time_placeholder = st.empty()
            def progress_callback(done, total, est_left):
                progress = done / total
                progress_placeholder.progress(progress, text=f"Running ANOVA: {done}/{total}")
                time_placeholder.info(f"Estimated time left: {int(est_left)} seconds")
            st.session_state.df_anova = anova(
                st.session_state.data,
                st.session_state.anova_attribute,
                corrections_map[st.session_state.p_value_correction],
                elements=st.session_state.anova_groups,
                _progress_callback=progress_callback
            )
            progress_placeholder.empty()
            time_placeholder.empty()
            st.rerun()

    # Ensure df_anova is initialized as an empty DataFrame if not already set
    st.session_state.setdefault("df_anova", pd.DataFrame())

    # Update the check to handle NoneType
    if st.session_state.df_anova is not None and not st.session_state.df_anova.empty:
        tukey_options = list(st.session_state.anova_groups) if "anova_groups" in st.session_state else []
        tukey_options.sort()
    
        prev_tukey_elements = st.session_state.get("_prev_tukeys_options", None)
        tukey_elements = c2.multiselect(
            "select **two** options for Tukey's comparison",
            options=tukey_options,
            default=tukey_options[:2],
            key="tukeys_options",
            max_selections=2,
            help="Tukey's HSD compares **pairs** of groups. Pick exactly 2.",
        )

        # if user changed groups, clear old results
        if prev_tukey_elements is not None and set(tukey_elements) != set(prev_tukey_elements):
            st.session_state.df_tukey = pd.DataFrame()
        st.session_state["_prev_tukeys_options"] = list(tukey_elements)
        st.session_state.tukey_elements = tukey_elements

        run_btn = c2.button(
            "Run Tukey's",
            key="run_tukey",
            type="primary",
            disabled=len(st.session_state.tukey_elements) != 2,
        )
        # Only use existing ANOVA results, NOT rerun ANOVA calculations
        if run_btn:
            import time
            progress_placeholder = st.empty()
            time_placeholder = st.empty()

            def progress_callback(done, total, est_left):
                progress = done / total
                progress_placeholder.progress(progress, text=f"Running Tukey's: {done}/{total}")
                if est_left is not None:
                    time_placeholder.info(f"Estimated time left: {int(est_left)} seconds")
            
            st.session_state.df_tukey = tukey(
                st.session_state.df_anova,
                st.session_state.anova_attribute,
                st.session_state.tukey_elements,
                corrections_map[st.session_state.p_value_correction],
                _progress_callback=progress_callback
            )

            progress_placeholder.empty()
            time_placeholder.empty()
            st.rerun()

    tab_options = [
        "📈 ANOVA: plot",
        "📁 ANOVA: result table",
        "📊 ANOVA: metabolites (boxplots)",
    ]

    if st.session_state.df_tukey is not None and not st.session_state.df_tukey.empty:
        tab_options += ["📈 Tukey's: plots", "📁 Tukey's: result table"]

    if st.session_state.df_anova is not None and not st.session_state.df_anova.empty:
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
            if ft is not None and not ft.empty:
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
            else:
                candidates = list(st.session_state.df_anova.index)

            candidates.sort()
            def metabolite_label(m):
                return str(m).split("&")[0] if "&" in str(m) else str(m)
            st.selectbox(
                "select metabolite",
                options=candidates,
                key="anova_metabolite",
                format_func=metabolite_label
            )

            met = st.session_state.anova_metabolite
            df_anova = st.session_state.df_anova

            # Show full metabolite name above the boxplot if available
            full_met_name = None

            if ft is not None and not ft.empty and met in ft.index and name_cols:
                name_col = name_cols[0]
                full_met_name = ft.at[met, name_col]
            if met in df_anova.index and "significant" in df_anova.columns:
                is_sig = df_anova.loc[met, "significant"]
                desc = "Significant" if is_sig else "Insignificant"
                if full_met_name:
                    st.write(f"**{desc} Metabolite: {full_met_name}**")
                else:
                    st.write(f"**{desc} Metabolite: {met}**")

            fig = get_metabolite_boxplot(
                st.session_state.df_anova,
                st.session_state.anova_metabolite,
            )

            show_fig(fig, f"anova-{st.session_state.anova_metabolite}")

        if hasattr(st.session_state, "df_tukey") and st.session_state.df_tukey is not None and not st.session_state.df_tukey.empty:
            with tabs[3]:
                fig1 = get_tukey_teststat_plot(getattr(st.session_state.df_tukey, '_original', st.session_state.df_tukey))
                show_fig(fig1, "tukeys-teststat")
                
                fig2 = get_tukey_volcano_plot(getattr(st.session_state.df_tukey, '_original', st.session_state.df_tukey))
                show_fig(fig2, "tukeys-volcano")
            
            with tabs[4]:
                df_tukey = st.session_state.df_tukey.copy()

                if "anova_total" in st.session_state:
                    st.caption(
                        f"ℹ️ Tukey's post-hoc was run on {len(df_tukey)} features "
                        f"out of {st.session_state.anova_total} ANOVA-tested features."
                    )

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
                    st.dataframe(df_tukey.style.format(style_dict), use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_tukey, use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ Please complete data preparation step first!")
