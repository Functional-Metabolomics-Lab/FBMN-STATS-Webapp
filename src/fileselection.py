import streamlit as st
from .common import *
from gnpsdata import taskresult
from gnpsdata import workflow_fbmn
import urllib
import io
import requests

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
    # drop only completely empty columns; a sample column with a few empty cells must not be lost
    ft = ft.dropna(axis=1, how="all")
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
                    "Could not determine index automatically, missing m/z and/or RT information in column names. Please upload a proper feature table or select the metabolite ID column manually."
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


# Everything Bagel (EB) runs its own feature finding, so its outputs live under
# nf_output/feature_finding/ instead of FBMN's nf_output/clustering/.
GNPS2_EB_QUANT_TABLE_PATHS = [ # MZmine-style feature table with per-sample "Peak area" columns
    "nf_output/feature_finding/feature_finding_results/aligned_features.csv",
]
GNPS2_EB_FEATURE_LIBRARY_PATHS = [ # feature library search results (annotations)
    "nf_output/feature_library_search/merged_feature_library_search_results.tsv",
]
GNPS2_EB_METADATA_PATHS = [ # fallback when no metadata file was uploaded with the task
    "nf_output/metadata/merged_metadata.tsv",
]
# Task parameters holding the uploaded metadata file (FBMN uses metadata_filename, EB uses metadata_file).
# GNPS2 stores the upload in a task input folder named after the parameter.
GNPS2_METADATA_PARAMS = ["metadata_filename", "metadata_file"]
GNPS2_EB_NETWORK_PATHS = [
    "nf_output/networking/filtered_pairs.tsv",
]
# An FBMN task launched downstream of EB keeps its own library results and networking,
# but reads its features from the upstream EB task (the "inputfeatures" parameter).
GNPS2_FBMN_LIBRARY_PATHS = [
    "nf_output/library/merged_results_with_gnps.tsv",
]


def _fetch_gnps2_dataframe_multi(task_id, candidate_paths, delimiter="\t"):
    """
    Tries each GNPS2 result path in priority order (on the prod, beta and de mirrors) and returns
    (dataframe, matched_path) for the first non-empty one, or (None, None) if none could be retrieved.
    Uses requests because GNPS2 rejects pandas' default urllib client with HTTP 403.
    """
    for path in candidate_paths:
        for server in ["prod", "beta", "de"]:
            try:
                url = taskresult.determine_gnps2_resultfile_url(task_id, path, gnps2server=server)
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                # A missing file returns the GNPS2 HTML page instead of an error status
                if "html" in resp.headers.get("content-type", ""):
                    continue
                df = pd.read_csv(io.BytesIO(resp.content), sep=delimiter)
                if not df.empty:
                    return df, path
            except Exception:
                continue
    return None, None


def _fetch_gnps2_task_params(task_id):
    """Returns the submission parameters of a GNPS2 task from its status.json ({} if unavailable)."""
    for server in ["prod", "beta", "de"]:
        try:
            base_url = taskresult.determine_gnps2_resultfile_url(task_id, "", gnps2server=server).split("/resultfile")[0]
            resp = requests.get(f"{base_url}/status.json", params={"task": task_id}, timeout=60)
            resp.raise_for_status()
            return resp.json().get("submission_parameters", {}) or {}
        except Exception:
            continue
    return {}


def _get_upstream_eb_features(params):
    """
    For an FBMN task run downstream of Everything Bagel, returns (eb_task_id, features_path)
    parsed from "inputfeatures" (TASKLOCATION/<eb_task_id>/<path>); otherwise (None, None).
    """
    input_features = params.get("inputfeatures", "")
    if params.get("featurefindingtool") == "EVERYTHING_BAGEL" and input_features.startswith("TASKLOCATION/"):
        parts = input_features.split("/", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    return None, None


def _fetch_gnps2_task_metadata(task_id, params):
    """
    Fetches the metadata file uploaded with a GNPS2 task from its input folder
    (e.g. metadata_filename/<uploaded file name>), falling back to GNPS2_EB_METADATA_PATHS.
    Returns (dataframe, matched_path) or (None, None).
    """
    for param in GNPS2_METADATA_PARAMS:
        uploaded = params.get(param)
        if uploaded:
            file_name = uploaded.split("/")[-1]
            delimiter = "," if file_name.lower().endswith(".csv") else "\t"
            md, md_path = _fetch_gnps2_dataframe_multi(task_id, [f"{param}/{file_name}"], delimiter=delimiter)
            if md is not None:
                return md, md_path

    return _fetch_gnps2_dataframe_multi(task_id, GNPS2_EB_METADATA_PATHS, delimiter="\t")


def load_from_gnps2_eb(task_id):

    """
    Returns (ft, md, an, nw) as DataFrames for an Everything Bagel GNPS2 task.
    - Accepts either the EB task ID or the ID of an FBMN task run downstream of EB. For the latter,
      the quantification table comes from the upstream EB task and everything else from the FBMN task.
    - EB is GNPS2-only, so there is no GNPS1 fallback.
    - The quantification table is required; metadata, annotations and node pairs are optional.
    """
    params = _fetch_gnps2_task_params(task_id)
    eb_task_id, upstream_features_path = _get_upstream_eb_features(params)
    if eb_task_id:
        st.write(f"↳ FBMN task downstream of Everything Bagel task `{eb_task_id}` — pulling the quantification table from there.")
        quant_task_id = eb_task_id
        quant_paths = [upstream_features_path] + GNPS2_EB_QUANT_TABLE_PATHS
    else:
        quant_task_id = task_id
        quant_paths = GNPS2_EB_QUANT_TABLE_PATHS

    # 1. Quantification table -- used as the signal that this is a reachable EB task.
    ft, quant_path = _fetch_gnps2_dataframe_multi(quant_task_id, quant_paths, delimiter=",")
    if ft is None:
        raise RuntimeError(
            f"❌ Failed to fetch data for Everything Bagel Task ID {task_id}. Tried quantification table path(s) "
            f"({', '.join(quant_paths)}) in task {quant_task_id}. Please double check the Task ID and that the "
            f"Everything Bagel workflow completed successfully on GNPS2."
        )
    st.write(f"✓ Successfully pulled Quantification Table (`{quant_path}`)")

    # 2. Metadata
    md, md_path = _fetch_gnps2_task_metadata(task_id, params)
    if isinstance(md, pd.DataFrame) and "filename" in md.columns:
        md = md.set_index("filename")
        st.write(f"✓ Successfully pulled Metadata (`{md_path}`)")
    else:
        md = pd.DataFrame()

    # 3. Annotations -- the quant table was found, so a failure here points at this specific step
    #    rather than a wrong task ID. A downstream FBMN task has its own library search results;
    #    otherwise use EB's feature library search results.
    an, an_path = None, None
    if eb_task_id:
        an, an_path = _fetch_gnps2_dataframe_multi(task_id, GNPS2_FBMN_LIBRARY_PATHS, delimiter="\t")
    if an is None:
        an, an_path = _fetch_gnps2_dataframe_multi(eb_task_id or task_id, GNPS2_EB_FEATURE_LIBRARY_PATHS, delimiter="\t")
    if an is None:
        st.warning(
            f"⚠️ Task {task_id} was found on GNPS2 (quantification table retrieved from `{quant_path}`), "
            f"but no library results could be retrieved. This can happen if no spectral library was selected "
            f"for this job, or if the task is still running. Continuing without annotations."
        )
        an = pd.DataFrame()
    else:
        st.write(f"✓ Successfully pulled Library Results (`{an_path}`)")
        # EB names differ from FBMN's library results; add the FBMN names so the existing merge works.
        # query_scan corresponds to the quantification table's "row ID".
        # Newer EB versions call the compound name column NAME instead of COMPOUND_NAME.
        for fbmn_col, eb_cols in [("#Scan#", ["query_scan"]), ("Compound_Name", ["COMPOUND_NAME", "NAME"]), ("MQScore", ["cosine"])]:
            eb_col = next((c for c in eb_cols if c in an.columns), None)
            if fbmn_col not in an.columns and eb_col is not None:
                an = an.assign(**{fbmn_col: an[eb_col]})

    # 4. Node pairs
    nw, _ = _fetch_gnps2_dataframe_multi(task_id, GNPS2_EB_NETWORK_PATHS, delimiter="\t")
    if nw is None:
        nw = pd.DataFrame()
    else:
        st.write("✓ Successfully pulled Node Pair Table")

    index_with_mz_RT = ft.apply(lambda x: f'{x["row ID"]}_{round(x["row m/z"], 4)}_{round(x["row retention time"], 2)}', axis=1)
    ft.index = index_with_mz_RT
    ft.index.name = 'metabolite'

    return ft, md, an, nw

### DISPLAY FUNCTIONS ###
def get_gnps_tables():

    ft = st.session_state.get("ft_gnps")
    md = st.session_state.get("md_gnps")
    an = st.session_state.get("an_gnps")
    nw = st.session_state.get("nw_gnps")
    merged = None
    name_key = None

    if "ft_gnps" in st.session_state and st.session_state["ft_gnps"] is not None:
        if hasattr(st.session_state["ft_gnps"], "empty") and not st.session_state["ft_gnps"].empty:
            ft = st.session_state["ft_gnps"]
    if "md_gnps" in st.session_state and st.session_state["md_gnps"] is not None:
        if hasattr(st.session_state["md_gnps"], "empty") and not st.session_state["md_gnps"].empty:
            md = st.session_state["md_gnps"]
    if "an_gnps" in st.session_state and st.session_state["an_gnps"] is not None:
        if hasattr(st.session_state["an_gnps"], "empty") and not st.session_state["an_gnps"].empty:
            an = st.session_state["an_gnps"]
    if "nw_gnps" in st.session_state and st.session_state["nw_gnps"] is not None:
        if hasattr(st.session_state["nw_gnps"], "empty") and not st.session_state["nw_gnps"].empty:
            nw = st.session_state["nw_gnps"]

    if ("an_gnps" in st.session_state and st.session_state["an_gnps"] is not None
        and hasattr(st.session_state["an_gnps"], "empty") and not st.session_state["an_gnps"].empty
        and ft is not None and hasattr(ft, 'merge')):
        an = st.session_state["an_gnps"]
        if 'row ID' in ft.columns and '#Scan#' in an.columns:
            ft['row ID'] = ft['row ID'].astype(str)
            an['#Scan#'] = an['#Scan#'].astype(str)
            name_key = "Compound_Name"
            merged = ft.merge(
                _one_annotation_per_key(an, "#Scan#"),
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
                display_dataframe(ft, title="quantification-table")
                st.write(f"Feature table shape: {ft.shape}")
            tab_index += 1
        if md_file in st.session_state and st.session_state[md_file] is not None and not st.session_state[md_file].empty:
            with tabs[tab_index]:
                display_dataframe(md, title="metadata-table")
                st.write(f"Metadata table shape: {md.shape}")
            tab_index += 1
        if an_file in st.session_state and st.session_state[an_file] is not None and not st.session_state[an_file].empty:
            with tabs[tab_index]:
                display_dataframe(an, title="annotation-table", hide_index=True)
                st.write(f"Annotation table shape: {an.shape}")
            tab_index += 1
        if nw_file in st.session_state and st.session_state[nw_file] is not None and not st.session_state[nw_file].empty:
            with tabs[tab_index]:
                display_dataframe(nw, title="node-pair-table", hide_index=True)
                st.write(f"Node pair table shape: {nw.shape}")
            tab_index += 1
        if ft_merged in st.session_state and st.session_state[ft_merged] is not None and not st.session_state[ft_merged].empty:
            with tabs[tab_index]:
                display_dataframe(merged, title="annotated-feature-table", hide_index=True)
                st.write(f"Annotated feature table shape: {merged.shape}")
            tab_index += 1


###MERGE FUNCTIONS###
def _one_annotation_per_key(an, key):
    """Keep a single annotation row per feature key (best MQScore if available).

    A left merge with duplicated keys would add rows to the feature table, which shifts
    compound names onto the wrong features when they are later matched by position.
    """
    if "MQScore" in an.columns:
        an = an.sort_values("MQScore", ascending=False, kind="stable")
    return an.drop_duplicates(subset=key, keep="first")


def merge_annotation(ft, an):
    # Handle None or empty DataFrames gracefully
    if ft is None or an is None or not hasattr(ft, 'columns') or not hasattr(an, 'columns') or ft.empty or an.empty:
        return pd.DataFrame(), None
    
    ft_columns = list(ft.columns)
    an_columns = list(an.columns)
    index1 = ft_columns.index("row ID") if "row ID" in ft_columns else 0
    index2 = an_columns.index("#Scan#") if "#Scan#" in an_columns else 0
    index3 = an_columns.index("Compound_Name") if "Compound_Name" in an_columns else 0
    
    col1, col2, col3 = st.columns(3)
    # Let user choose the join keys (one from each table)
    ft_key = col1.selectbox("Column in feature table to merge", list(ft.columns), key="merge_ft_key", index=index1)
    an_key = col2.selectbox("Column in annotation table to merge", list(an.columns), key="merge_an_key", index=index2)
    name_key = col3.selectbox("Name column in annotation table", list(an.columns), key="name_an_key", index=index3)

    # Coerce keys to string and do a LEFT merge (keep only IDs present in ft)
    ft_merge = ft.copy()
    an_merge = an.copy()
    ft_merge[ft_key] = ft_merge[ft_key].astype(str)
    an_merge[an_key] = an_merge[an_key].astype(str)
    an_merge = _one_annotation_per_key(an_merge, an_key)

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

