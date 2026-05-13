import streamlit as st
import pandas as pd
import io
import uuid
import base64

from src.chat import render_sidebar_chat

dataframe_names = ("md",
                   "data",
                   "ft",
                   "an",
                   "nw",
                   "ft_with_annotations",
                   "df_anova",
                   "df_tukey",
                   "df_rm_anova",
                   "df_ttest",
                   "df_kruskal",
                   "df_dunn",
                   "df_important_features",
                   "df_oob",
                   "ft_gnps",
                   "md_gnps",
                   "an_gnps",
                   "nw_gnps",
                   "df_gnps_annotations",
                   "df_wilcoxon",
                   "df_friedman")

corrections_map = {"no correction": "none",
                   "Benjamini/Hochberg FDR": "fdr_bh",
                   "Sidak": "sidak",
                   "Bonferroni": "bonf",
                   "Benjamini/Yekutieli FDR": "fdr_by",
                   }

def reset_dataframes():
    for key in dataframe_names:
        st.session_state[key] = None

def init_state():
    defaults = {
        "task_id": "",
        "data_preparation_done": False,
        "ft_uploaded": None,
        "md_uploaded": None,
        "ft_gnps": None,
        "md_gnps": None,
        "chat_history": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def clear_cache_button():
    if st.button("Clear Cache"):
        # Clear cache for both newer and older Streamlit versions
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
        
        # Clear all session state variables
        for key in dataframe_names:
            st.session_state[key] = None
        
        # Reset chat history
        if "chat_history" in st.session_state:
            st.session_state["chat_history"] = []

        st.success("Cache cleared!")

def page_setup():
    # streamlit configs
    st.set_page_config(
        page_title="Statistics for Metabolomics",
        page_icon="assets/app_icon.ico",
        # layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    # initialize global session state variables if not already present
    # DataFrames
    for key in dataframe_names:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()
    if "data_preparation_done" not in st.session_state:
        st.session_state["data_preparation_done"] = False

    with st.sidebar:
        # AI chat assistant (shown above settings)
        render_sidebar_chat()

        with st.expander("⚙️ Settings", expanded=True):
            st.selectbox("p-value correction",
                         corrections_map.keys(),
                         key="p_value_correction")
            st.selectbox(
                "image export format",
                ["svg", "png", "jpeg", "webp"],
                key="image_format",
            )

            # Add the clear cache button
            v_space(1)
            clear_cache_button()

        # Display two images side by side in the sidebar
        v_space(1)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<a href="https://github.com/Functional-Metabolomics-Lab/FBMN-STATS" target="_blank">'
                f'<img src="data:image/png;base64,{base64.b64encode(open("assets/FBMN-STATS-GUIed_logo2.png", "rb").read()).decode()}" width="130">'
                '</a>',
                unsafe_allow_html=True,
            )
        with col2:
             st.markdown(
                f'<a href="https://gnps2.org/homepage" target="_blank">'
                f'<img src="data:image/png;base64,{base64.b64encode(open("assets/GNPS2_logo.png", "rb").read()).decode()}" width="150">'
                '</a>',
                unsafe_allow_html=True,
            )
            
           # st.image("assets/GNPS2_logo.png", use_column_width=True)

        v_space(1)

        #st.image("assets/vmol-icon.png", use_column_width=True)
        st.markdown(
                f'<a href="https://vmol.org/ " target="_blank">'
                f'<img src="data:image/png;base64,{base64.b64encode(open("assets/vmol-icon.png", "rb").read()).decode()}" width="300">'
                '</a>',
                unsafe_allow_html=True,
            )
        v_space(1)

        st.markdown("## Functional-Metabolomics-Lab")
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            """<a href="https://github.com/Functional-Metabolomics-Lab">
            <img src="data:image/png;base64,{}" width="50">
            </a>""".format(
                base64.b64encode(open("./assets/github-logo.png", "rb").read()).decode()
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            """<a href="https://www.functional-metabolomics.com/">
            <img src="data:image/png;base64,{}" width="50">
            </a>""".format(
                base64.b64encode(open("./assets/fmlab_logo_colored.png", "rb").read()).decode()
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            """<a href="https://www.youtube.com/@functionalmetabolomics">
            <img src="data:image/png;base64,{}" width="50">
            </a>""".format(
                base64.b64encode(open("./assets/youtube-logo.png", "rb").read()).decode()
            ),
            unsafe_allow_html=True,
        )

        v_space(1)

def v_space(n, col=None):
    for _ in range(n):
        if col:
            col.write("")
        else:
            st.write("")

def open_df(file):
    separators = {"txt": "\t", "tsv": "\t", "csv": ","}
    try:
        if type(file) == str:
            ext = file.split(".")[-1]
            if ext != "xlsx":
                df = pd.read_csv(file, sep=separators[ext])
            else:
                df = pd.read_excel(file)
        else:
            ext = file.name.split(".")[-1]
            if ext != "xlsx":
                df = pd.read_csv(file, sep=separators[ext])
            else:
                df = pd.read_excel(file)
        # sometimes dataframes get saved with unnamed index, that needs to be removed
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", inplace=True, axis=1)
        
        # Fix data type issues that cause Arrow serialization problems
        df = _fix_dataframe_types(df)
        
        return df
    except:
        return pd.DataFrame()

def _fix_dataframe_types(df):
    """
    Fix data type issues in dataframes that cause Arrow serialization and hashing problems.
    Converts problematic object columns to string or numeric types.
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric first
            try:
                # Check if the column contains mostly numeric values
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                non_null_count = numeric_values.count()
                total_count = len(df[col])
                
                # If more than 80% of values are numeric, convert to numeric
                if non_null_count / total_count > 0.8:
                    df[col] = numeric_values
                else:
                    # Convert to string to avoid mixed types
                    df[col] = df[col].astype(str).replace('nan', '')
            except:
                # If conversion fails, convert to string
                df[col] = df[col].astype(str).replace('nan', '')
    
    # Remove any columns that might contain unhashable types like numpy arrays
    for col in df.columns:
        try:
            # Test if the column can be hashed (this will catch numpy arrays)
            pd.util.hash_pandas_object(df[[col]])
        except (TypeError, ValueError):
            # Convert problematic columns to string representation
            df[col] = df[col].apply(lambda x: str(x) if x is not None else '')
    
    return df

def show_table(df, title="", col="", download=True, hide_index=False):
    if col:
        col = col
    else:
        col = st
    col.dataframe(df, use_container_width=True, hide_index=hide_index)

    # Persist shown tables keyed by page so the LLM chat can access them
    current_page = st.session_state.get("current_page", "")
    if "_shown_tables" not in st.session_state:
        st.session_state["_shown_tables"] = []
    # Replace any existing entry for the same page+title (fresh on each rerun)
    st.session_state["_shown_tables"] = [
        t for t in st.session_state["_shown_tables"]
        if not (t["page"] == current_page and t["title"] == title)
    ]
    st.session_state["_shown_tables"].append({"page": current_page, "title": title, "df": df})
    # Cap total stored entries to avoid memory bloat
    if len(st.session_state["_shown_tables"]) > 20:
        st.session_state["_shown_tables"] = st.session_state["_shown_tables"][-20:]

    return df

def show_fig(fig, download_name, container_width=True):
    # Store last figure in session for chat context
    st.session_state["last_figure"] = fig

    st.plotly_chart(
        fig,
        use_container_width=container_width,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "zoom",
                "pan",
                "select",
                "lasso",
                "zoomin",
                "autoscale",
                "zoomout",
                "resetscale",
            ],
            "toImageButtonOptions": {
                "filename": download_name,
                "format": st.session_state.image_format,
            },
        },
    )


def filter_top_significant_points_ui(df, key_prefix, *, min_n=5, max_n=100, default_n=25):
    """Optionally filter a results DataFrame to the top-N most significant rows.

    Significance ranking is based on the first available p-value column in:
    p-corrected, p-val, p, stats_p.
    """
    if df is None or len(df) == 0:
        return df

    checkbox_key = f"{key_prefix}_top_sig_enabled"
    slider_key = f"{key_prefix}_top_sig_n"

    st.checkbox(
        "Show top most significant points",
        value=False,
        key=checkbox_key,
        help="When checked, show only the most significant points on the plot.",
    )

    if not st.session_state.get(checkbox_key, False):
        return df

    p_col = next((c for c in ["p-corrected", "p_val", "p-val", "p", "stats_p"] if c in df.columns), None)
    if p_col is None:
        st.info("No p-value column was found to rank significance. Showing all points.")
        return df

    total_rows = len(df)
    slider_max = min(max_n, total_rows)
    if slider_max <= 0:
        return df

    slider_min = min_n if slider_max >= min_n else 1
    slider_default = min(default_n, slider_max)
    if slider_default < slider_min:
        slider_default = slider_min

    top_n = st.slider(
        "Number of most significant points to show",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_default,
        key=slider_key,
    )

    ranked = df.copy()
    ranked[p_col] = pd.to_numeric(ranked[p_col], errors="coerce")
    ranked = ranked.sort_values(p_col, ascending=True, na_position="last").head(top_n)
    st.caption(f"Showing top {len(ranked)} points ranked by {p_col}.")
    return ranked

def download_plotly_figure(fig, filename="", col=""):
    buffer = io.BytesIO()
    fig.write_image(file=buffer, format="png")

    if col:
        col.download_button(
            label=f"Download Figure",
            data=buffer,
            file_name=filename,
            mime="application/png",
        )
    else:
        st.download_button(
            label=f"Download Figure",
            data=buffer,
            file_name=filename,
            mime="application/png",
        )


def _get_current_page_name():
    """Best-effort retrieval of the current page name.

    Individual pages can optionally set `st.session_state["current_page"]` for
    more accurate labeling.
    """

    return st.session_state.get("current_page") or st.session_state.get("page_name") or "Current analysis page"
