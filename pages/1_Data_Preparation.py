import streamlit as st
import pandas as pd
from src.common import *
from src.fileselection import *
from src.cleanup import *

page_setup()
 
st.markdown("# File Selection")

if st.session_state["data_preparation_done"]:
    st.success("Data preparation was successful!")
    if st.button("Re-do the data preparation step now."):
        reset_dataframes()
        st.session_state["data_preparation_done"] = False
        st.rerun()
    show_table(pd.concat([st.session_state.md, st.session_state.data], axis=1), title="FeatureMatrix-scaled-centered")
else:
    st.info(
        """💡 Once you are happy with the results, don't forget to click the **Submit Data for Statistics!** button."""
    )

    ft, md, an, nw = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    file_origin = st.radio("File origin", 
                           ["Quantification table and meta data files", 
                            "GNPS(2) FBMN task ID", 
                            "Example dataset from publication", 
                            "Small example dataset for testing",
                            "GNPS2 classical molecular networking (CMN) task ID"])
    
    #Initialize keys
    for k in ["ft_gnps", "md_gnps", "an_gnps", "nw_gnps", "ft_with_annotations"]:
        st.session_state.setdefault(k, pd.DataFrame())
    
    # b661d12ba88745639664988329c1363e
    if file_origin == "Small example dataset for testing":
        ft, md = load_example()
        ft = ft.set_index('metabolite')
        an, nw, ft_with_annotations = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  
        st.session_state.update({"ft": ft, "md": md, "an": an, "nw": nw, "ft_with_annotations": ft_with_annotations})
        show_all_files_in_table("ft", "md", "an", "nw", "ft_with_annotations")

    elif file_origin in ["GNPS(2) FBMN task ID", "Example dataset from publication", "GNPS2 classical molecular networking (CMN) task ID"]:
        
        st.warning("💡 This tool only supports FBMN task ID from GNPS1 and 2 not from Quickstart GNPS1.")
        
        if file_origin == "Example dataset from publication":
            task_id_default = "b661d12ba88745639664988329c1363e" # 63e8b3da08df41fe95031e4710e0476b
            disabled = True
            task_id = st.text_input("GNPS FBMN task ID", task_id_default, disabled=disabled, help="This is a GNPS1 FBMN task ID used in the publication.")
            
            _, c2, _ = st.columns(3)
            if c2.button("Load files from GNPS", type="primary", disabled=len(task_id) == 0, use_container_width=True):
                st.session_state["ft_gnps"], st.session_state["md_gnps"], st.session_state["an_gnps"], st.session_state["nw_gnps"] = load_from_gnps_fbmn(task_id)
                st.session_state["ft_gnps"] = st.session_state["ft_gnps"]
                
                ft, md, an, nw, merged, name_key = get_gnps_tables()
                st.session_state["ft_with_annotations"] = merged
                st.session_state["name_key"] = name_key
            show_all_files_in_table("ft_gnps", "md_gnps", "an_gnps", "nw_gnps", "ft_with_annotations")            
        
        elif file_origin == "GNPS2 classical molecular networking (CMN) task ID":
            if st.session_state.get("ft_with_annotations") is not None:
                st.session_state["ft_with_annotations"] = None

            task_id_default = "" # 2a65f90094654235a4c8d337fdca11e1
            disabled = False

            task_id = st.text_input("GNPS2 CMN task ID", task_id_default, disabled=disabled)
            if task_id:
                st.session_state["task_id"] = task_id
            else:
                st.session_state["task_id"] = None

            _, c2, _ = st.columns(3)
            
            if c2.button("Load files from GNPS2", type="primary", disabled=len(task_id) == 0, use_container_width=True):
                st.session_state["ft_gnps"], st.session_state["md_gnps"],  st.session_state["an_gnps"], st.session_state["nw_gnps"] = load_from_gnps2_cmn(task_id)
                st.session_state["ft_gnps"] = st.session_state["ft_gnps"]

                if not st.session_state["ft_gnps"].empty and st.session_state["md_gnps"].empty:
                    st.warning("⚠️ **Metadata file is missing.** The metadata is essential for performing statistical analysis and understanding the context of your data. Please upload one.")
                    md_file = st.file_uploader("Metadata Table")
                    if md_file:
                        md = load_md(md_file)
                        st.success("Metadata was loaded successfully!")

                ft, md, an, nw, merged, name_key = get_gnps_tables()
                st.session_state["ft_with_annotations"] = merged
                st.session_state["name_key"] = name_key

            elif task_id is None or len(task_id) == 0:
                st.session_state["ft_gnps"] = None
                st.session_state["md_gnps"] = None
                st.session_state["an_gnps"] = None
                st.session_state["nw_gnps"] = None

            show_all_files_in_table("ft_gnps", "md_gnps", "an_gnps", "nw_gnps", "ft_with_annotations")
        

        elif file_origin == "GNPS(2) FBMN task ID":
            task_id_default = ""
            disabled = False

            task_id = st.text_input("FBMN task ID", task_id_default, disabled=disabled, help="GNPS1 or GNPS2 FBMN task ID")
            if task_id:
                st.session_state["task_id"] = task_id
            else:
                st.session_state["task_id"] = None
                st.session_state["ft_gnps"] = None
                st.session_state["md_gnps"] = None
                st.session_state["an_gnps"] = None
                st.session_state["nw_gnps"] = None
                st.session_state["ft_with_annotations"] = None
            
            _, c2, _ = st.columns(3)
            
            if c2.button("Load files from GNPS(2)", type="primary", disabled=len(task_id) == 0, use_container_width=True):
                st.session_state["ft_gnps"], st.session_state["md_gnps"], st.session_state["an_gnps"], st.session_state["nw_gnps"] = load_from_gnps_fbmn(task_id)
                st.session_state["ft_gnps"] = st.session_state["ft_gnps"]

                if not st.session_state["ft_gnps"].empty and st.session_state["md_gnps"].empty:
                    st.warning("⚠️ **Metadata file is missing.** The metadata is essential for performing statistical analysis and understanding the context of your data. Please upload one.")
                    md_file = st.file_uploader("Metadata Table")
                    if md_file:
                        md = load_md(md_file)
                        st.success("Metadata was loaded successfully!")
                
                ft, md, an, nw, merged, name_key = get_gnps_tables()
                st.session_state["ft_gnps"] = ft
                st.session_state["md_gnps"] = md
                st.session_state["an_gnps"] = an
                st.session_state["nw_gnps"] = nw
                st.session_state["ft_with_annotations"] = merged
                st.session_state["name_key"] = name_key
            
            elif task_id is None or len(task_id) == 0:
                st.session_state["ft_gnps"] = None
                st.session_state["md_gnps"] = None
                st.session_state["an_gnps"] = None
                st.session_state["nw_gnps"] = None

            show_all_files_in_table("ft_gnps", "md_gnps", "an_gnps", "nw_gnps", "ft_with_annotations")
        
        st.session_state["ft"] = st.session_state["ft_gnps"]
        st.session_state["md"] = st.session_state["md_gnps"]
        st.session_state["an"] = st.session_state["an_gnps"]
        st.session_state["nw"] = st.session_state["nw_gnps"]
        st.session_state["ft_with_annotations"] = st.session_state.get("ft_with_annotations", pd.DataFrame())
        
    if file_origin == "Quantification table and meta data files":
        if st.session_state.get("ft_with_annotations") is not None:
            st.session_state["ft_with_annotations"] = None
        
        st.info("💡 Upload tables in txt (tab separated), tsv, csv or xlsx (Excel) format.")
        c1, c2 = st.columns(2)
        
        # Feature Quantification Table
        ft_file = c1.file_uploader("Quantification Table", 
                                   type=["csv", "xlsx", "txt", "tsv"],
                                   help = ("Output of mass spectrometry studies. The table shows a list of mass spectral features with their relative intensities " 
                                   "(represented by its integrated peak area) across samples."
                                   ),
                                   key="ft_uploader",
                                   )
     
        # Meta Data Table
        md_file = c2.file_uploader("Meta Data Table", 
                                   type=["csv", "xlsx", "txt", "tsv"],
                                   help = ("User-created table providing more context for the samples (e.g., sample type, species, and tissue)"),
                                   key="md_uploader",
                                   )
        
        # Create 2 columns for the nw, annotation file uploaders
        c3, c4 = st.columns(2)
        
        # Annotation file uploader
        annotation_file = c3.file_uploader("Annotation Table (Optional)", 
                                           type=["csv", "xlsx", "txt", "tsv"],
                                           key="an_uploader",
                                   )

        #Node Pair File uploader
        nw_file = c4.file_uploader("Node Pair Table (Optional)", 
                                   type=["csv", "xlsx", "txt", "tsv"],
                                   help = (
                                       "E.g: Molecular network edges: each row is a node pairs(Cluster IDs 1 & 2) with "
                                       "cosine score, m/z difference, etc. that can be used as edge connecting the nodes."
                                   ),
                                   key="nw_uploader",
                                   )

        # --- Load into session_state only when a new file arrives ---
        if ft_file:
            st.session_state["ft_uploaded"] = load_ft(ft_file).set_index('metabolite')
        else:
            st.session_state["ft_uploaded"] = None
        if md_file:
            st.session_state["md_uploaded"] = load_md(md_file)
        else:
            st.session_state["md_uploaded"] = None
        if annotation_file:
            if st.session_state.get("ft_uploaded") is None or st.session_state.get("md_uploaded") is None:
                st.error("Feature table or Metadata table is missing. Please upload both before uploading the annotation table.")
            else:
                st.session_state['an_uploaded'] = load_annotation(annotation_file)
        else:
            st.session_state['an_uploaded'] = None
        if nw_file:
            st.session_state['nw_uploaded'] = load_nw(nw_file)
        else:
            st.session_state['nw_uploaded'] = None
       
        # --- Retrieve from session_state ---
        if st.session_state["ft_uploaded"] is not None or st.session_state["md_uploaded"] is not None:
            ft, md, an, nw = get_uploaded_tables()
        
        st.session_state["ft"] = ft
        st.session_state["md"] = md
        st.session_state["an"] = an
        st.session_state["nw"] = nw
        show_all_files_in_table("ft", "md", "an", "nw", "ft_with_annotations")
        st.session_state["ft_with_annotations"] = st.session_state.get("ft_with_annotations", pd.DataFrame())

        # --- Collect messages ---
        messages = []
        if st.session_state["ft"] is None:
            messages.append(("error", "Please upload a valid **feature quantification table**!"))
        if st.session_state["md"] is None:
            messages.append(("error", "Please upload a valid **metadata table**!"))
        if st.session_state["an"] is None:
            messages.append(("info", "No annotation table provided. Continuing without annotations."))
        if st.session_state["nw"] is None:
            messages.append(("info", "No node pair table provided. Continuing without node pairs."))

        # --- Display messages after ---
        for level, msg in messages:
            if level == "error":
                st.error(msg)
            elif level == "info":
                st.info(msg)
    
    if ft is not None and not ft.empty and not ft.index.is_unique:
        st.error("Please upload a feature matrix with unique metabolite names.")
    if ft is not None and not ft.empty and md is not None and not md.empty:
        st.success("Files loaded successfully!")

    st.markdown("# Data Cleanup")
    
    with st.expander("📖 About"):
        st.markdown("**Removal of blank features**")
        st.image("assets/figures/blank-removal.png")
        st.markdown("**Imputation of missing values**")
        st.image("assets/figures/imputation.png")
        st.markdown("**Data scaling and centering**")
        st.image("assets/figures/scaling.png")
    
    # Check if the data is available in the session state
    if ('ft' in st.session_state and 'md' in st.session_state and 
        st.session_state['ft'] is not None and not st.session_state['ft'].empty and 
        st.session_state['md'] is not None and not st.session_state['md'].empty
    ):

        ft = st.session_state.get('ft').copy()
        md = st.session_state.get('md').copy()

        # clean up meta data table
        md = clean_up_md(md)

        # clean up feature table and remove unneccessary columns
        ft = clean_up_ft(ft)

        # # check if ft column names and md row names are the same
        md, ft = check_columns(md, ft)

        # Initialize the process flags at the start of your Streamlit app if they don't already exist
        if 'blank_removal_done' not in st.session_state:
            st.session_state['blank_removal_done'] = False

        if 'imputation_done' not in st.session_state:
            st.session_state['imputation_done'] = False

        # Use a string to track the normalization method used; 'None' indicates no normalization done
        if 'normalization_method_used' not in st.session_state:
            st.session_state['normalization_method_used'] = 'None'

        tabs = st.tabs(["**Blank Removal**", "**Imputation**", "**Normalization**", "📊 **Summary**"])
        with tabs[0]:
            blank_removal = st.checkbox("Remove blank features?", False)
            if blank_removal:
                # Select true sample files (excluding blank and pools)
                st.markdown("#### Samples")
                st.markdown(
                    "Select samples (excluding blank and pools) based on the following table."
                )
                df = inside_levels(md)
                mask = df.apply(lambda row: len(row['LEVELS']) == 0, axis=1)
                df = df[~mask]
                st.dataframe(df)
                c1, c2 = st.columns(2)
                sample_column = c1.selectbox(
                    "attribute for sample selection",
                    md.columns,
                )
                sample_options = list(set(md[sample_column].dropna()))
                sample_rows = c2.multiselect("sample selection", sample_options, sample_options[0])
                samples = ft[md[md[sample_column].isin(sample_rows)].index]
                samples_md = md.loc[samples.columns]

                with st.expander(f"Selected samples preview (n={samples.shape[1]})"):
                    st.dataframe(samples.head())

                if samples.shape[1] == ft.shape[1]:
                    st.warning("You selected everything as sample type. Blank removal not possible.")
                else:
                    v_space(1)
                    # Ask if blank removal should be done
                    st.markdown("#### Blanks")
                    st.markdown(
                        "Select blanks (excluding samples and pools) based on the following table."
                    )
                    non_samples_md = md.loc[
                        [index for index in md.index if index not in samples.columns]
                    ]
                    df = inside_levels(non_samples_md)
                    mask = df.apply(lambda row: len(row['LEVELS']) == 0, axis=1)
                    df = df[~mask]
                    st.dataframe(df)
                    c1, c2 = st.columns(2)

                    blank_column = c1.selectbox(
                        "attribute for blank selection", non_samples_md.columns
                    )
                    blank_options = list(set(non_samples_md[blank_column].dropna()))
                    blank_rows = c2.multiselect("blank selection", blank_options, blank_options[0])
                    blanks = ft[non_samples_md[non_samples_md[blank_column].isin(blank_rows)].index]
                    with st.expander(f"Selected blanks preview (n={blanks.shape[1]})"):
                        st.dataframe(blanks.head())

                    # define a cutoff value for blank removal (ratio blank/avg(samples))
                    c1, c2 = st.columns(2)
                    cutoff = c1.number_input(
                        "cutoff threshold for blank removal",
                        0.1,
                        1.0,
                        0.3,
                        0.05,
                        help="""The recommended cutoff range is between 0.1 and 0.3.
                        Features with intensity ratio of (blank mean)/(sample mean) above the threshold (e.g. 30%) are considered noise/background features.
                        """,
                    )
                    (
                        ft,
                        n_background_features,
                        n_real_features,
                    ) = remove_blank_features(blanks, samples, cutoff)
                    c2.metric("background or noise features", n_background_features)
                    with st.expander(f"Feature table after removing blanks - features: {ft.shape[0]}, samples: {ft.shape[1]}"):
                        show_table(ft, "blank-features-removed")
            
                st.session_state['blank_removal_done'] = True
            else:
                st.session_state['blank_removal_done'] = False
            
            if not ft.empty:
                cutoff_LOD = get_cutoff_LOD(ft)

                with tabs[1]:

                    c1, c2 = st.columns(2)
                    c2.metric(
                        f"total missing values",
                        str((ft == 0).to_numpy().mean() * 100)[:4] + " %",
                    )
                    imputation = c1.checkbox("Impute missing values?", False, help=f"These values will be filled with random number between 1 and {cutoff_LOD} (Limit of Detection) during imputation.")
                    if imputation:
                        if cutoff_LOD > 1:
                            c1, c2 = st.columns(2)
                            ft = impute_missing_values(ft, cutoff_LOD)
                            with st.expander(f"Imputed data - features: {ft.shape[0]}, samples: {ft.shape[1]}"):
                                show_table(ft.head(), "imputed")
                        else:
                            st.warning(f"Can't impute with random values between 1 and lowest value, which is {cutoff_LOD} (rounded).")
                        
                        st.session_state['imputation_done'] = True
                    else:
                        st.session_state['imputation_done'] = False

                with tabs[2]:
                    normalization_method = st.radio("data normalization method", ["None",
                                                            "Center-Scaling", 
                                                            # "Probabilistic Quotient Normalization (PQN)", 
                                                            "Total Ion Current (TIC) or sample-centric normalization"])
                    st.session_state['normalization_method_used'] = normalization_method
                
                with tabs[3]:
                    # Summary tab content
                    st.markdown("## Process Summary")
                    if st.session_state['blank_removal_done']:
                        st.success("Blank removal done.")
                    else:
                        st.warning("Blank removal not done.")

                    if st.session_state['imputation_done']:
                        st.success("Imputation done.")
                    else:
                        st.warning("Imputation not done.")

                    # Check which normalization method was used
                    if st.session_state['normalization_method_used'] != 'None':
                        st.success(f"Normalization done using {st.session_state['normalization_method_used']} method.")
                    else:
                        st.warning("Normalization not done.")

                    tab1, tab2 = st.tabs(
                        ["📊 Feature intensity frequency", "📊 Missing values per feature"]
                    )
                    with tab1:
                        fig = get_feature_frequency_fig(ft)
                        show_fig(fig, "feature-intensity-frequency")
                    with tab2:
                        fig = get_missing_values_per_feature_fig(ft, cutoff_LOD)
                        show_fig(fig, "missing-values")
            else:
                st.error("No features left after blank removal!")
        
        _, c1, _ = st.columns(3)
        if c1.button("**Submit Data for Statistics!**", type="primary"):
            st.session_state["md"], st.session_state["data"] = normalization(
                ft, md, normalization_method
            )
            st.session_state["data_preparation_done"] = True
            st.rerun()

            # # Display all rows of the "metabolite" column
            # if "metabolite" in st.session_state.data.columns:
            #     st.dataframe(st.session_state.data["metabolite"])
            # else:
            #     st.warning("The 'metabolite' column is not present in the data.")

