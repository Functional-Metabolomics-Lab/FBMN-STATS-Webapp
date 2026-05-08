import streamlit as st
from src.common import *
from src.kruskal import *

page_setup()
st.session_state["current_page"] = "Kruskal-Wallis & Dunn's"

st.markdown("# Kruskal Wallis & Dunn's post hoc")

with st.expander("📖 About"):
    st.markdown(
        """
        The **Kruskal–Wallis (KW) test** is a non-parametric alternative to one-way ANOVA.
        It checks whether there are statistically significant differences **among three or more groups** without assuming normal distribution or equal variances. 
        If the KW test indicates significant differences among groups, the user can apply Dunn's post-hoc test to directly compare a specific pair of groups to identify where those differences occur.
         
        The first figure block displays the **K-statistic** and p-value calculation for all groups, showing whether at least one group differs. 
        The second block shows the concept of performing **Dunn’s post hoc test** on the significant features from KW, highlighting *which groups* are driving the differences.
        
        💡 *Tip:*  Use KW and Dunn’s tests when data are **non-normal**, **heteroscedastic**, or **ordinal**. If your data are normally distributed and have equal variances, use **ANOVA** instead.
        """
        )
    
    st.image("assets/figures/kruskal-wallis.png")
   
if st.session_state.data is not None and not st.session_state.data.empty:

    # Ensure df_kruskal and df_dunn are initialized
    st.session_state.setdefault("df_kruskal", pd.DataFrame())
    st.session_state.setdefault("df_dunn", pd.DataFrame())

    # Build top-level tabs: Dunn's only appears after KW has been run
    top_tab_labels = ["Kruskal-Wallis"]
    if st.session_state.df_kruskal is not None and not st.session_state.df_kruskal.empty:
        top_tab_labels.append("Dunn's")

    top_tabs = st.tabs(top_tab_labels)

    # ───────────────────────── KRUSKAL-WALLIS TAB ─────────────────────────
    with top_tabs[0]:
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
            st.session_state.pop("kruskal_attempted_metabolites", None)
            st.session_state.pop("kruskal_returned_metabolites", None)
            st.session_state.pop("dunn_attempted_metabolites", None)
            st.session_state.pop("dunn_returned_metabolites", None)

        st.session_state["_prev_kruskal_attribute"] = kruskal_attribute

        attribute = st.session_state.kruskal_attribute
        attribute_options = list(set(st.session_state.md[attribute].dropna()))
        attribute_options.sort()

        prev_kruskal_groups = st.session_state.get("_prev_kruskal_groups", None)
        kruskal_groups = c2.multiselect(
            "Select groups to include in Kruskal Wallis (minimum 3)",
            options=attribute_options,
            default=attribute_options,
            key="kruskal_groups",
            help="For comparing 2 groups, use the t-test page instead.  If button is disabled, select a different attribute.",
        )

        if prev_kruskal_groups is not None and set(kruskal_groups) != set(prev_kruskal_groups):
            st.session_state.df_kruskal = pd.DataFrame()
            st.session_state.df_dunn = pd.DataFrame()
            st.session_state.pop("kruskal_attempted_metabolites", None)
            st.session_state.pop("kruskal_returned_metabolites", None)
            st.session_state.pop("dunn_attempted_metabolites", None)
            st.session_state.pop("dunn_returned_metabolites", None)
        st.session_state["_prev_kruskal_groups"] = list(kruskal_groups)

        min_required = 3
        run_disabled = not ("kruskal_groups" in st.session_state and len(st.session_state.kruskal_groups) >= min_required)

        st.button("Run Kruskal Wallis", key="run_kruskal", type="primary", disabled=run_disabled)

        if st.session_state.run_kruskal:
            if "kruskal_groups" not in st.session_state or len(st.session_state.kruskal_groups) < min_required:
                st.error(f"At least {min_required} groups must be selected to run Kruskal Wallis.")
            else:
                import time
                progress_placeholder = st.empty()
                time_placeholder = st.empty()
                start_time = time.time()
                def progress_callback(done, total, est_left):
                    progress = done / total
                    elapsed = time.time() - start_time
                    if done > 0:
                        est_total = elapsed / done * total
                        est_left = est_total - elapsed
                    else:
                        est_left = 0
                    progress_placeholder.progress(progress, text=f"Running Kruskal-Wallis: metabolite {done} of {total}")
                    time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")
                st.session_state.df_kruskal = kruskal_wallis(
                    st.session_state.data,
                    st.session_state.kruskal_attribute,
                    corrections_map[st.session_state.p_value_correction],
                    elements=st.session_state.kruskal_groups,
                    _progress_callback=progress_callback
                )
                progress_placeholder.empty()
                time_placeholder.empty()
                st.rerun()

        kruskal_attempted = st.session_state.get("kruskal_attempted_metabolites")
        kruskal_returned = st.session_state.get("kruskal_returned_metabolites")
        if kruskal_attempted is not None and kruskal_returned is not None and kruskal_attempted != kruskal_returned:
            st.warning(
                f"Kruskal-Wallis attempted {kruskal_attempted} metabolites, but only {kruskal_returned} produced plottable results. "
                f"{kruskal_attempted - kruskal_returned} metabolite(s) were skipped because valid test outputs could not be computed for the selected filters."
            )

        # KW result sub-tabs
        if st.session_state.df_kruskal is not None and not st.session_state.df_kruskal.empty:
            kw_sub_tabs = st.tabs([
                "📈 KW: plot",
                "📁 KW: result table",
                "📊 KW: metabolites (boxplots)",
            ])

            with kw_sub_tabs[0]:
                color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
                st.selectbox("Color significant points by", options=color_by_options, key="kruskal_color_by")
                _kw_color = st.session_state.get("kruskal_color_by", "Significance (default)")
                kw_plot_df = filter_top_significant_points_ui(st.session_state.df_kruskal, "kruskal_plot")
                fig = get_kruskal_plot(kw_plot_df, color_by=None if _kw_color == "Significance (default)" else _kw_color)
                show_fig(fig, "kruskal")
                st.session_state["page_figs_kw_plot"] = fig

            with kw_sub_tabs[1]:
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

            with kw_sub_tabs[2]:
                # Include both significant and insignificant metabolites in dropdown
                all_metabolites = sorted(list(st.session_state.df_kruskal["metabolite"]))
                def metabolite_label(m):
                    return str(m).split("&")[0] if "&" in str(m) else str(m)
                st.selectbox(
                    "select metabolite",
                    all_metabolites,
                    key="kruskal_metabolite",
                    format_func=metabolite_label
                )

                met = st.session_state.kruskal_metabolite
                df_kruskal = st.session_state.df_kruskal
                # Show full metabolite name above the boxplot if available
                full_met_name = None
                # Try to get full name from ft_gnps if available
                ft = st.session_state.get("ft_gnps", None)
                if ft is not None and not ft.empty and met in ft.index:
                    name_cols = [c for c in ("metabolite_name", "name", "feature_name", "compound_name", "compound") if c in ft.columns]
                    if name_cols:
                        name_col = name_cols[0]
                        full_met_name = ft.at[met, name_col]
                if met in df_kruskal.index and "significant" in df_kruskal.columns:
                    is_sig = df_kruskal.loc[met, "significant"]
                    desc = "Significant" if is_sig else "Insignificant"
                    if full_met_name:
                        st.write(f"**{desc} Metabolite: {full_met_name}**")
                    else:
                        st.write(f"**{desc} Metabolite: {met}**")

                fig = get_metabolite_boxplot(
                    st.session_state.df_kruskal,
                    st.session_state.kruskal_metabolite,
                )

                show_fig(fig, f"kruskal-{st.session_state.kruskal_metabolite}")
                st.session_state["page_figs_kw_boxplot"] = fig

    # ───────────────────────── DUNN'S TAB ─────────────────────────
    if st.session_state.df_kruskal is not None and not st.session_state.df_kruskal.empty:
        with top_tabs[1]:
            dunn_options = list(st.session_state.kruskal_groups) if "kruskal_groups" in st.session_state else []
            dunn_options.sort()

            c1d, c2d = st.columns(2)

            prev_kruskal_elements = st.session_state.get("_prev_dunns_options", None)
            dunn_elements = c1d.multiselect(
                "select **two** options for Dunn's comparison",
                options=dunn_options,
                default=dunn_options[:2],
                key="dunn_elements",
                max_selections=2,
                help="Select two options.",
            )

            if prev_kruskal_elements is not None and set(dunn_elements) != set(prev_kruskal_elements):
                st.session_state.df_dunn = pd.DataFrame()
                st.session_state.pop("dunn_attempted_metabolites", None)
                st.session_state.pop("dunn_returned_metabolites", None)
            st.session_state["_prev_dunns_options"] = list(dunn_elements)

            c1d.button(
                "Run Dunn's",
                key="run_dunn",
                type="primary",
                disabled=len(st.session_state.dunn_elements) != 2,
            )

            if st.session_state.run_dunn:
                import time
                progress_placeholder = st.empty()
                time_placeholder = st.empty()
                start_time = time.time()
                def progress_callback(done, total, est_left):
                    progress = done / total
                    elapsed = time.time() - start_time
                    if done > 0:
                        est_total = elapsed / done * total
                        est_left = est_total - elapsed
                    else:
                        est_left = 0
                    progress_placeholder.progress(progress, text=f"Running Dunn's test: metabolite {done} of {total}")
                    time_placeholder.info(f"Estimated time remaining: {int(est_left)} seconds")
                st.session_state.df_dunn = dunn(
                    st.session_state.df_kruskal,
                    st.session_state.kruskal_attribute,
                    st.session_state.dunn_elements,
                    corrections_map[st.session_state.p_value_correction],
                    _progress_callback=progress_callback
                )
                progress_placeholder.empty()
                time_placeholder.empty()
                st.rerun()

            dunn_attempted = st.session_state.get("dunn_attempted_metabolites")
            dunn_returned = st.session_state.get("dunn_returned_metabolites")
            if dunn_attempted is not None and dunn_returned is not None and dunn_attempted != dunn_returned:
                st.warning(
                    f"Dunn's attempted {dunn_attempted} metabolites, but only {dunn_returned} produced plottable results. "
                    f"{dunn_attempted - dunn_returned} metabolite(s) were skipped because valid test outputs could not be computed for the selected filters."
                )

            # Dunn result sub-tabs
            if st.session_state.df_dunn is not None and not st.session_state.df_dunn.empty:
                dunn_sub_tabs = st.tabs(["📈 Dunn's: plots", "📁 Dunn's: result table"])

                with dunn_sub_tabs[0]:
                    color_by_options = ["Significance (default)"] + sorted([c for c in st.session_state.md.columns if len(set(st.session_state.md[c])) > 1])
                    st.selectbox("Color significant points by", options=color_by_options, key="dunn_color_by")
                    dunn_numeric = getattr(st.session_state.df_dunn, "_original", st.session_state.df_dunn)
                    # Test-statistic style plot (x = rank_sum_diff, y = -log10(p))
                    _dunn_color = st.session_state.get("dunn_color_by", "Significance (default)")
                    fig1 = get_dunn_teststat_plot(dunn_numeric, color_by=None if _dunn_color == "Significance (default)" else _dunn_color)
                    show_fig(fig1, "dunn-teststat")
                    st.session_state["page_figs_dunn_teststat"] = fig1
                    # Volcano plot (x = logFC, y = -log10(p))
                    fig2 = get_dunn_volcano_plot(dunn_numeric)
                    show_fig(fig2, "dunn-volcano")
                    st.session_state["page_figs_dunn_volcano"] = fig2

                with dunn_sub_tabs[1]:
                    df_dunn = st.session_state.df_dunn.copy()
                    df_dunn = df_dunn.rename(columns={
                        "stats_metabolite": "metabolite",
                        "stats_significant": "significant"
                    })

                    attribute_value = (
                        st.session_state.kruskal_attribute
                        if "kruskal_attribute" in st.session_state
                        else ""
                    )
                    dunn_groups = (
                        st.session_state.dunn_elements
                        if "dunn_elements" in st.session_state
                        else ["", ""]
                    )
                    if len(dunn_groups) == 2:
                        A_value, B_value = dunn_groups
                    else:
                        A_value, B_value = "", ""

                    # insert meta cols right after "significant"
                    sig_idx = (
                        df_dunn.columns.get_loc("significant") + 1
                        if "significant" in df_dunn.columns
                        else len(df_dunn.columns)
                    )

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

                    if "data" in st.session_state and "dunn_n" in st.session_state:
                        st.caption(
                            f"ℹ️ Dunn's post-hoc was run on {st.session_state.dunn_n} features "
                            f"out of {st.session_state.data.shape[1]} KW-tested features."
                        )

                    if style_dict:
                        styled = df_dunn.style.format(style_dict)
                        st.dataframe(styled, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(df_dunn, use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ Please complete data preparation step first!")
