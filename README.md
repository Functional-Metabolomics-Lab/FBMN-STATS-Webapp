![banner](assets/FBMN-STATS-GUIed_logo2.png)

[![Open in Statistics App!](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fbmn-stats.streamlit.app/)

A web app implementation of the [statistics notebooks](https://github.com/Functional-Metabolomics-Lab/Statistical-analysis-of-non-targeted-LC-MSMS-data) for metabolomics by the [Functional Metabolomics Lab](https://github.com/Functional-Metabolomics-Lab). These notebooks are developed by the Virtual Multi Omics Lab ([VMOL](https://vmol.org/)).

## Installation
- [run the app](https://metabolomics-statistics.streamlit.app/) without installation (recommended for smaller datasets)

### Local Installation
You can run the Statistics for Metabolomics app in two ways:

1. **From source code**
   - Download or clone this repository.
   - Open a terminal in the project folder and run:
     ```bash
     pip install -r requirements.txt
     streamlit run Statistics_for_Metabolomics.py
     ```
   - The app will open automatically in your browser.

2. **Using the Windows executable (.msi)**
   - Download the latest `.msi` version from the releases page.
   - ⚠️ Before installing a new version, **delete all files from any older installation** (for example, from `C:/`) or (Control Panel >> Programs >> Programs & Features >> Uninstall a Program >> remove FBMN-STATS-GUIed)
   - This helps avoid compatibility or version conflicts so the app runs smoothly.
   - Once downloading, if the pop-up terminal seems blank or isn't loading, restart the app. 

## Available Statistics
- Principal Component Analysis (PCA)
- Multivariate
    - PERMANOVA & PCoA
- Hierachical Clustering & Heatmaps
- Univariate 
    - One-way ANOVA & Tukey's post hoc test
    - Kruskal-Wallis & Dunn's post hoc test
- Student's t-test

## Quickstart

Once you have completed the **Data Preparation** step, chose any of the available statistics sections.

### Data Preparation
- two tables are required: **Quantification** and **Meta Data**
- supported formats: `tsv` and `txt` (tab separated), `csv` (comma separated) and `xlsx` (Excel file)
- if feature table has an optional **metabolite** column that will be taken as index (can be unique ID, contain `m/z` and `RT` information or actual metabolite name)
- feature index can be automatically generated if columns for `m/z` and `RT` (and optionally `row ID`) are present
- sample file names need to contain `mzML` file name extensions
- quantification table needs sample file names as column names
- meta data table **requires** a `filename` column
- meta data table can contain columns with attributes
- checkout the **example data** availabe in file selection
- remove blank features and impute missing values in the **Data Cleanup** section

Example feature table:
|metabolite|sample1.mzML|sample2.mzML|blank.mzML|
|---|---|---|---|
|1|1000|1100|100|
|2|2000|2200|200|

Example meta data table:
|filename|Sample_Type|Time_Point|
|---|---|---|
|sample1.mzML|Sample|1h|
|sample2.mzML|Sample|2h|
|blank.mzML|Blank|N/A|
