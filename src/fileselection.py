import streamlit as st
from .common import *
from gnpsdata import taskresult
from gnpsdata import workflow_fbmn
import urllib

patterns = [
    ["m/z", "mz", "mass over charge"],
    ["rt", "retention time", "retention-time", "retention_time"],
]


def string_overlap(string, options):
    for option in options:
        if option in string and "mzml" not in string:
            return True
    return False


def get_new_index(df):
    # get m/z values (cols[0]) and rt values (cols[1]) column names
    cols = [
        [col for col in df.columns.tolist() if string_overlap(col.lower(), pattern)]
        for pattern in patterns
    ]
    try:
        # select the first match for each
        column_names = [col[0] for col in cols if col]
        if not column_names:
            return df, "no matching columns"
        # set metabolites column as first column (not index)
        df["metabolite"] = df.index
        if len(column_names) == 2:
            df["metabolite"] = df[column_names[0]].round(5).astype(str)
            if column_names[1]:
                df["metabolite"] = (
                    df["metabolite"] + "@" + df[column_names[1]].round(2).astype(str)
                )
            if "row ID" in df.columns:
                df["metabolite"] = df["row ID"].astype(str) + "_" + df["metabolite"]
        # Move 'metabolite' to the first column
        cols_order = ["metabolite"] + [col for col in df.columns if col != "metabolite"]
        df = df[cols_order]
    except Exception:
        return df, "fail"
    return df, "success"


allowed_formats = "Allowed formats: csv (comma separated), tsv (tab separated), txt (tab separated), xlsx (Excel file)."


def load_example():
    ft = open_df("example-data/FeatureMatrix.csv")
    ft, _ = get_new_index(ft)
    md = open_df("example-data/MetaData.txt").set_index("filename")
    return ft, md

def load_ft(ft_file):
    ft = open_df(ft_file)
    ft = ft.dropna(axis=1)
    # determining index with m/z, rt and adduct information

    if "metabolite" in ft.columns:
        ft.index = ft["metabolite"]
    else:
        
        if st.checkbox("Create index automatically", 
                       value=True,
                       help=(
        "No column named 'metabolite' was found. "
        "You can either manually choose another column as the metabolite ID, "
        "or let the app automatically create a unique index based on ID, m/z, and RT."
    )):
            ft, msg = get_new_index(ft)
            if msg == "no matching columns":
                st.warning(
                    "Could not determine index automatically, missing m/z and/or RT information in column names."
                )
        else:
            metabolite_col = st.selectbox(
                "Column with unique values to use as metabolite ID.",
                list(ft.columns),
                help=(
                    "Select any column from the table to use as the metabolite ID. "
                    "Make sure to choose a column that uniquely identifies each metabolite."
                    )
            )
            if metabolite_col:
                ft = ft.rename(columns={metabolite_col: "metabolite"})
                ft.index = ft["metabolite"].astype(str)
                ft = ft.drop(columns=["metabolite"])
    if ft.empty:
        st.error(f"Check quantification table!\n{allowed_formats}")
    return ft


def load_md(md_file):
    md = open_df(md_file)
    # we need file names as index, if they don't exist throw a warning and let user chose column
    if "filename" in md.columns:
        md.set_index("filename", inplace=True)
    else:
        v_space(2)
        st.warning(
            """⚠️ **Meta Data Table**

No 'filename' column for samples specified.

Please select the correct one."""
        )
        filename_col = st.selectbox("Column to use for sample file names.", [col for col in md.columns if len(md[col]) == len(set(md[col]))])
        if filename_col:
            md = md.set_index(filename_col)
            md.index = md.index.astype(str)
    if md.empty:
        st.error(f"Check meta data table!\n{allowed_formats}")

    return md

def load_annotation(annotation_file):
    """
    Load and process annotation file 
    """
    an_gnps = open_df(annotation_file)
    return an_gnps

def load_nw(network_file):
    """
    Load and process network pair file. 
    """
    nw = open_df(network_file)
    return nw


### GNPS LOADING FUNCTIONS ###
def load_from_gnps_fbmn(task_id):

    """
    - Returns (ft, md, an, nw) as DataFrames.
    - FBMN (cmn=False): tries GNPS2 API first, falls back to GNPS1 URLs.
    """
    ft = md = an = nw = None

     # -------- Special case: default task id --------
    if task_id == "b661d12ba88745639664988329c1363e":
        return load_from_gnps1_fbmn(task_id)

    # --------Normal worfflow: Try GNPS2 FBMN API --------
    try: # GNPS2 will run here
        ft = workflow_fbmn.get_quantification_dataframe(task_id, gnps2=True)
        md = workflow_fbmn.get_metadata_dataframe(task_id, gnps2=True)
        if isinstance(md, pd.DataFrame) and "filename" in md.columns:
            md = md.set_index("filename")
        
        an = taskresult.get_gnps2_task_resultfile_dataframe(task_id, "nf_output/library/merged_results_with_gnps.tsv")
        nw = taskresult.get_gnps2_task_resultfile_dataframe(task_id, "nf_output/networking/filtered_pairs.tsv")
            
        # Force fallback if feature table is missing/empty
        if ft is None or (isinstance(ft, pd.DataFrame) and ft.empty):
            raise ValueError("Empty result from GNPS2 — falling back to GNPS1.")
                
    except (urllib.error.HTTPError, ValueError, AttributeError, KeyError) as e:
        st.error(f"GNPS2 unavailable or empty: {e}") # GNPS1 task IDs can not be retrieved and throw HTTP Error 500
        return load_from_gnps1_fbmn(task_id)

    if not isinstance(md, pd.DataFrame): # Handle empty metadata
        md = pd.DataFrame()
    if not isinstance(an, pd.DataFrame):
        an = pd.DataFrame()
    if not isinstance(nw, pd.DataFrame):
        nw = pd.DataFrame()

    index_with_mz_RT = ft.apply(lambda x: f'{x["row ID"]}_{round(x["row m/z"], 4)}_{round(x["row retention time"], 2)}', axis=1)
    ft.index = index_with_mz_RT
    ft.index.name = 'metabolite'

    return ft, md, an, nw

def load_from_gnps1_fbmn(task_id: str):

    """Load FBMN tables from GNPS1 URLs. Always returns DataFrames (ft required, others optional)."""
    """Returns (ft, md, an, nw) as DataFrames."""

    ft_url = f"https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile?task={task_id}&file=quantification_table_reformatted/&block=main"
    md_url = f"https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile?task={task_id}&file=metadata_merged/&block=main"
    an_url = f"https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile?task={task_id}&file=DB_result/&block=main"
    nw_url = f"https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile?task={task_id}&file=networking_pairs_results_file_filtered/&block=main"
    try:
        ft = pd.read_csv(ft_url)
    except Exception as e:
        raise RuntimeError(
            f"❌ Could not load feature table from GNPS1 for task {task_id}.\n"
            f"Possible reasons: Mistyped Task ID, Task is not an FBMN job, GNPS server unreachable.\n"
            f"Error: {e}"
            )
        
    try:
        md = pd.read_csv(md_url, sep="\t", index_col="filename")
    except (urllib.error.HTTPError, FileNotFoundError, KeyError, pd.errors.EmptyDataError) as e:
        st.warning(f"Warning: could not load metadata file: {e}")
        md = pd.DataFrame()
            
    try:
        an = pd.read_csv(an_url, sep="\t")
    except (pd.errors.EmptyDataError, FileNotFoundError, KeyError) as e:
        st.warning(f"Warning: could not load annotation file: {e}")
        an = pd.DataFrame()

    try:
        nw = pd.read_csv(nw_url, sep="\t")
    except (pd.errors.EmptyDataError, FileNotFoundError, KeyError) as e:
        st.warning(f"Warning: could not load network file: {e}")
        nw = pd.DataFrame()
    
    index_with_mz_RT = ft.apply(lambda x: f'{x["row ID"]}_{round(x["row m/z"], 4)}_{round(x["row retention time"], 2)}', axis=1)
    ft.index = index_with_mz_RT
    ft.index.name = 'metabolite'
    
    return ft, md, an, nw


def load_from_gnps2_cmn(task_id):

    """
    Returns (ft, md, an, nw) as DataFrames for CMN.
    - cmn True; an/nw often unavailable -> empty DFs.
    """
    ft = md = an = nw = None
    
    try: # GNPS2 will run here
         ft_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/clustering/featuretable_reformatted_precursorintensity.csv"
         md_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/metadata/merged_metadata.tsv" 
         an_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/library/merged_results_with_gnps.tsv"
         nw_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/networking/filtered_pairs.tsv"
         
         try:
             ft = pd.read_csv(ft_url)
         except Exception as e:
             st.error(f"Failed to load CMN feature table: {e}")
             ft = None
             
         try:
             md = pd.read_csv(md_url, sep = "\t", index_col="filename")
         except pd.errors.EmptyDataError:
             md = pd.DataFrame()

         try:
             an = pd.read_csv(an_url, sep="\t")
         except pd.errors.EmptyDataError:
             an = pd.DataFrame()

         try:
             nw = pd.read_csv(nw_url, sep="\t")
         except pd.errors.EmptyDataError:
             nw = pd.DataFrame()
    
    except (urllib.error.HTTPError, ValueError) as e:
        print(f"HTTP Error encountered: {e}") # GNPS1 CMN task IDs can not be retrieved and throw HTTP Error 500
    
    if ft is None or ft.empty:
        raise ValueError("Empty result from workflow_fbmn — falling back to CMN CSV path.")
            
    if not isinstance(md, pd.DataFrame): # Handle empty metadata
        md = pd.DataFrame()
    if not isinstance(an, pd.DataFrame):
        an = pd.DataFrame()
    if not isinstance(nw, pd.DataFrame):
        nw = pd.DataFrame()

    index_with_mz_RT = ft.apply(lambda x: f'{x["row ID"]}_{round(x["row m/z"], 4)}_{round(x["row retention time"], 2)}', axis=1)
    ft.index = index_with_mz_RT
    ft.index.name = 'metabolite'
    ft = ft.drop(columns=["row m/z", "row retention time"])
    
    return ft, md, an, nw
 
### DISPLAY FUNCTIONS ###
def get_gnps_tables():

    ft = st.session_state.get("ft_gnps")
    md = st.session_state.get("md_gnps")
    an = st.session_state.get("an_gnps")
    nw = st.session_state.get("nw_gnps")

    if "ft_gnps" in st.session_state:
        if not st.session_state["ft_gnps"].empty:
            ft = st.session_state["ft_gnps"]
   
    if "md_gnps" in st.session_state:
        if not st.session_state["md_gnps"].empty:
            md = st.session_state["md_gnps"]

    if "an_gnps" in st.session_state:
        if not st.session_state["an_gnps"].empty:
            an = st.session_state["an_gnps"]
    
    if "nw_gnps" in st.session_state:
        if not st.session_state["nw_gnps"].empty:
            nw = st.session_state["nw_gnps"]

    if st.session_state["an_gnps"] is not None and not st.session_state["an_gnps"].empty:
        an = st.session_state["an_gnps"]
        ft['row ID'] = ft['row ID'].astype(str)
        an['#Scan#'] = an['#Scan#'].astype(str)
        name_key = "Compound_Name"

        # Merge on index vs. #Scan# column
        merged = ft.merge(
            an, 
            left_on="row ID",
            right_on="#Scan#",
            how="left",
            suffixes=("", "_an"))
    else:
        an = None

    return ft, md, an, nw, merged, name_key

def get_uploaded_tables():
    ft = st.session_state.get("ft_uploaded")
    md = st.session_state.get("md_uploaded")
    an = st.session_state.get("an_uploaded")
    nw = st.session_state.get("nw_uploaded")

    if "ft_uploaded" in st.session_state:
        if st.session_state["ft_uploaded"] is not None and not st.session_state["ft_uploaded"].empty:
            ft = st.session_state["ft_uploaded"]
    else:
        ft = pd.DataFrame()

    if "md_uploaded" in st.session_state:
        if st.session_state["md_uploaded"] is not None and not st.session_state["md_uploaded"].empty:
            md = st.session_state["md_uploaded"]
    else:
        md = pd.DataFrame()

    if "an_uploaded" in st.session_state:
        if st.session_state["an_uploaded"] is not None and not st.session_state["an_uploaded"].empty:
            an = st.session_state["an_uploaded"]

            merged, name_key = merge_annotation(ft, an)
            st.session_state["ft_with_annotations"] = merged
            st.session_state["name_key"] = name_key
        else:
            an = pd.DataFrame()
    else:
        an = pd.DataFrame()

    if "nw_uploaded" in st.session_state:
        if st.session_state["nw_uploaded"] is not None and not st.session_state["nw_uploaded"].empty:
            nw = st.session_state["nw_uploaded"]
        else:
            nw = pd.DataFrame()
    else:
        nw = pd.DataFrame()

    return ft, md, an, nw


def show_all_files_in_table(ft_file, md_file, an_file, nw_file, ft_merged):
    tab_options = []

    if ft_file in st.session_state:
        if st.session_state[ft_file] is not None and not st.session_state[ft_file].empty:
            ft = st.session_state[ft_file]
            tab_options.append("**Quantification Table**")

    if md_file in st.session_state:
        if st.session_state[md_file] is not None and not st.session_state[md_file].empty:
            md = st.session_state[md_file]
            tab_options.append("**Metadata Table**")

    if an_file in st.session_state:
        if st.session_state[an_file] is not None and not st.session_state[an_file].empty:
            an = st.session_state[an_file]
            tab_options.append("**Annotation Table**")

    if nw_file in st.session_state:
        if st.session_state[nw_file] is not None and not st.session_state[nw_file].empty:
            nw = st.session_state[nw_file]
            tab_options.append("**Node Pair Table**")
    
    if ft_merged in st.session_state:
        if st.session_state[ft_merged] is not None and not st.session_state[ft_merged].empty:
            merged = st.session_state[ft_merged]
            tab_options.append("**Annotated Feature Table**")

    if tab_options: # and task_id:
        tabs = st.tabs(tab_options)
        tab_index = 0
        if ft_file in st.session_state and st.session_state[ft_file] is not None and not st.session_state[ft_file].empty:
            with tabs[tab_index]:
                st.dataframe(ft, hide_index=False)
                st.write(f"Feature table shape: {ft.shape}")
            tab_index += 1
        if md_file in st.session_state and st.session_state[md_file] is not None and not st.session_state[md_file].empty:
            with tabs[tab_index]:
                st.dataframe(md, hide_index=False)
                st.write(f"Metadata table shape: {md.shape}")
            tab_index += 1
        if an_file in st.session_state and st.session_state[an_file] is not None and not st.session_state[an_file].empty:
            with tabs[tab_index]:
                st.dataframe(an, hide_index=True)
                st.write(f"Annotation table shape: {an.shape}")
            tab_index += 1
        if nw_file in st.session_state and st.session_state[nw_file] is not None and not st.session_state[nw_file].empty:
            with tabs[tab_index]:
                st.dataframe(nw, hide_index=True)
                st.write(f"Node pair table shape: {nw.shape}")
            tab_index += 1
        if ft_merged in st.session_state and st.session_state[ft_merged] is not None and not st.session_state[ft_merged].empty:
            with tabs[tab_index]:
                st.dataframe(merged, hide_index=True)
                st.write(f"Annotated feature table shape: {merged.shape}")
            tab_index += 1


###MERGE FUNCTIONS###
def merge_annotation(ft, an):
    col1, col2, col3 = st.columns(3)
    # Let user choose the join keys (one from each table)
    ft_key = col1.selectbox("Column in feature table to merge", list(ft.columns), key="merge_ft_key")
    an_key = col2.selectbox("Column in annotation table to merge", list(an.columns), key="merge_an_key")
    name_key = col3.selectbox("Name column in annotation table", list(an.columns), key="name_an_key")

    # Coerce keys to string and do a LEFT merge (keep only IDs present in ft)
    ft_merge = ft.copy()
    an_merge = an.copy()
    ft_merge[ft_key] = ft_merge[ft_key].astype(str)
    an_merge[an_key] = an_merge[an_key].astype(str)

    merged = ft_merge.merge(
        an_merge,
        left_on=ft_key,
        right_on=an_key,
        how="left",          # only IDs from feature table
        suffixes=("", "_an")) # avoid column name clashes
    
    return merged, name_key

def merge_annotation_gnps(ft, an):
    # Coerce keys to string and do a LEFT merge (keep only IDs present in ft)
    ft_merge = ft.copy()
    an_merge = an.copy()
    ft_merge.index = ft_merge.index.astype(str)
    an_merge['#Scan#'] = an_merge['#Scan#'].astype(str)

    name_key = "Compound_Name"

    # Merge on index vs. #Scan# column
    merged = ft_merge.merge(
        an_merge,
        left_index=True,
        right_on="#Scan#",
        how="left",
        suffixes=("", "_an")
    )
    return merged, name_key

##################

