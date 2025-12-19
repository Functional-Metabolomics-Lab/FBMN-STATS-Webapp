import streamlit as st
import pandas as pd
import io
import uuid
import base64
from src.chat import openaiassistantchat, openai4ochat, gemini_chat

dataframe_names = ("md",
                   "data",
                   "ft",
                   "an",
                   "nw",
                   "ft_with_annotations",
                   "df_anova",
                   "df_tukey",
                   "df_ttest",
                   "df_kruskal",
                   "df_dunn",
                   "df_important_features",
                   "df_oob",
                   "ft_gnps",
                   "md_gnps",
                   "an_gnps",
                   "nw_gnps",
                   "df_gnps_annotations")

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
        # add more keys your app depends on
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

        st.success("Cache cleared!")

def page_setup(page_name):
    # streamlit configs
    st.set_page_config(
        page_title="Statistics for Metabolomics",
        page_icon="assets/app_icon.ico",
        # layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    # Set the page streamlit variable
    st.session_state["page"] = page_name

    # Set a global unique chat ID for the session
    if "chat_id" not in st.session_state:
        st.session_state["chat_id"] = str(uuid.uuid4())

    # initialize global session state variables if not already present
    # DataFrames
    for key in dataframe_names:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()
    if "data_preparation_done" not in st.session_state:
        st.session_state["data_preparation_done"] = False

    with st.sidebar:
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

        with st.container(height=750):
            with st.expander("💬 Chat", expanded=True):
                gemini_chat()
                # openai4ochat()
            
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

def show_table(df, title="", col="", download=True):
    if col:
        col = col
    else:
        col = st
    col.dataframe(df, use_container_width=True)

def show_fig(fig, download_name, container_width=True):
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
