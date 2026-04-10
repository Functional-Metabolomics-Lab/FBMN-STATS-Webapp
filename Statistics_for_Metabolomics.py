import streamlit as st
import base64
from src.common import *

page_setup()
init_state()
st.session_state["current_page"] = "Home"

st.image("assets/FBMN-STATS-GUIed_logo2.png", use_container_width=True)

st.markdown("""
## Quickstart Guide

Welcome to the FBMN-STATS, a web app implementation of the [statistics notebooks](https://github.com/Functional-Metabolomics-Lab/Statistical-analysis-of-non-targeted-LC-MSMS-data) for metabolomics by the [Functional Metabolomics Lab](https://github.com/Functional-Metabolomics-Lab) 
            as part of the article '[The Statistical Analysis of Feature-based Molecular Networking Results from Non-Targeted Metabolomics Data](https://doi.org/10.26434/chemrxiv-2023-wwbt0)'. 
            These notebooks are developed by the Virtual MultiOmics Lab ([VMOL](https://vmol.org/)).
            This app facilitates downstream statistical analysis of molecular networking data, simplifying the process for researchers.       

### Getting Started
This web app requires two primary inputs: a feature quantification table and metadata table. 
            Users with FBMN Job IDs from GNPS or GNPS2 can easily fetch necessary files by entering the job IDs via the 'Data Preparation' page. 
            This page also offers subsequent data cleanup steps such as Blank Removal, Imputation, and Normalization.
            """)

st.warning("""**⚠️** Our data cleanup options, including TIC normalization and center-scaling, are provided for quick and educational use. 
            These methods may not suit all datasets, so users should choose normalization strategies that best fit their data. 
            For more detailed guidance, please refer to our article or reach out to us to suggest additional methods you’d find useful.
""")

st.subheader('Data Preparation Essentials')
st.info("""
**Required Inputs & Supported Formats**
- Must upload two tables: **Feature quantification table** and **Metadata**
- Supported file types: **.tsv**, **.txt**, **.csv**, and **.xlsx**
- Feature tables should include a **`metabolite`** column for direct indexing
- Automatic feature indexing is supported via **`m/z`**, **`RT`**, and **`row ID`** columns
- Example datasets are available for quick reference
""")

st.warning("""
**⚠️ Important Notes**
- Sample filenames must contain the **`mzML`** extension for correct mapping
- The metadata table must include a **`filename`** column; additional attribute columns are optional
- Ensure consistent sample names between feature and metadata tables
- Proceed to **Data Cleanup** for blank removal and missing value imputation
""")

st.markdown("""          
Example feature table:  
 
|metabolite|sample1.mzML|sample2.mzML|blank.mzML|
|---|---|---|---|
|1|1000|1100|100|
|2|2000|2200|200|
""")
st.write(' ')

st.markdown("""        
Example meta data table:
            
|filename|Sample_Type|Time_Point|
|---|---|---|
|sample1.mzML|Sample|1h|
|sample2.mzML|Sample|2h|
|blank.mzML|Blank|N/A| 
""")

st.subheader('Statistical Analyses Available')
st.markdown("""
            - **Unsupervised Learning:** PCA, PCoA (with PERMANOVA), Hierarchical Clustering & Heatmaps  
            - **Supervised Learning:** Random Forest (simplified implementation)  
            - **Univariate Analysis:**  
                - **ANOVA** and Tukey's test (parametric, > 2 groups)  
                - **Student's t-test** (parametric, for 2 groups)  
                - **Kruskal–Wallis test** and Dunn's test (non-parametric, > 2 groups)  
                - **Mann-Whitney U test** (non-parametric, 2 independent groups)  
                - **Wilcoxon Signed-Rank test** (non-parametric, 2 paired groups)  
                - **Friedman test** (non-parametric, ≥ 3 paired groups)
""")

st.subheader('Outputs')
st.markdown("""
            Generated results include csv tables and images, aiding in data presentation and publication.
            """)

st.subheader('Interactive Plots')
st.markdown("""
💡 **All plots are interactive!**
- Select area with your mouse to zoom in
- Double click in the plot to zoom back out
- Save plots using the camera icon in the top right corner (specify image format in settings panel)
""")

st.subheader('Integrated AI Assistance')
st.markdown("""
            For users new to metabolomics or statistical analysis, we have integrated an AI assistant to provide guidance and answer questions. 
            View the AI assistant in the sidebar and feel free to ask any questions related to the app. 
            The assistant can help with understanding statistical concepts, interpreting results, and providing recommendations for next steps in your analysis.
""")
            


st.subheader('Settings Panel')
st.markdown("""
1. **P-value Correction:** These are FDR (False Disovery Rate) corrections applied for multiple univariate tests. Available options include Bonferroni, Sidak, Benjamini-Hochberg (BH), Benjamini-Yekutieli (BY), and an option for no correction.  
            While Bonferroni is known for controlling false positives, it may inadvertently increase false negatives. Advanced methods like BH and BY aim to balance true discoveries against false positives more effectively. We recommend BH for FDR correction to optimize analysis outcomes.
            
    Note that changing the p-value correction settings does not automatically update the corrected p-values. To update results re-run the analysis.

2. **Image Export Format:** Choose from svg, png, jpeg, webp. Recommendations: png for presentations, svg for publications.
""")


st.subheader('Limitations of the App')
st.markdown("""
- Be mindful of the 200 MB data limit. There is a possibility of server slowdowns or crashes with larger datasets. The GUI's simplicity and limited graphical options are designed for introductory purposes and does not allow customization for deeper analysis. 
""")


st.subheader('Citation and Resources')
st.markdown("""
For citations and further resources, please refer to our [article](https://doi.org/10.26434/chemrxiv-2023-wwbt0). You can download the Windows executables of the app from our [homepage](https://www.functional-metabolomics.com/resources).
""")


st.subheader("We Value Your Feedback")
st.markdown("""
Your feedback and suggestions are invaluable to us as we strive to enhance this tool. 
            As a small team, we greatly appreciate any contributions you can make. 
            Please feel free to create an issue on our GitHub repository to share your thoughts or report any issues you encounter.

[Create an Issue on GitHub](https://github.com/Functional-Metabolomics-Lab/FBMN-STATS-Webapp/issues/new)
""")

st.subheader("Functional-Metabolomics-Lab")

c1, c2, c3 = st.columns(3)
c1.markdown(
    """<a href="https://github.com/Functional-Metabolomics-Lab/FBMN-STATS-Webapp">
    <img src="data:image/png;base64,{}" width="100">
    </a>""".format(
        base64.b64encode(open("./assets/github-logo.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    """<a href="https://www.functional-metabolomics.com/">
    <img src="data:image/png;base64,{}" width="100">
    </a>""".format(
        base64.b64encode(open("./assets/fmlab_logo_colored.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    """<a href="https://www.youtube.com/@functionalmetabolomics">
    <img src="data:image/png;base64,{}" width="100">
    </a>""".format(
        base64.b64encode(open("./assets/youtube-logo.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)

