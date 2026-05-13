import pandas as pd
import streamlit as st
import io


def generate_boxplot_pdf_generic(df, metabolites, boxplot_fn):
    """Generate a PDF with 4 boxplots per page (2×2 grid).

    Parameters
    ----------
    df : pd.DataFrame
        The test-result DataFrame (metabolites as index) passed to boxplot_fn.
    metabolites : list
        Ordered list of metabolite identifiers.
    boxplot_fn : callable
        Function with signature ``(df, metabolite) -> plotly.Figure``.

    Returns
    -------
    bytes
        Raw PDF bytes ready for ``st.download_button``.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    page_w, page_h = A4
    margin = 36
    col_gap = 10
    row_gap = 10
    cols, rows_per_page = 2, 2
    per_page = cols * rows_per_page

    img_w = (page_w - 2 * margin - (cols - 1) * col_gap) / cols
    img_h = (page_h - 2 * margin - (rows_per_page - 1) * row_gap) / rows_per_page
    render_w = int(img_w * 2)
    render_h = int(img_h * 2)

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    for i, metabolite in enumerate(metabolites):
        pos = i % per_page
        if pos == 0 and i > 0:
            c.showPage()

        fig = boxplot_fn(df, metabolite)
        png_bytes = fig.to_image(format="png", width=render_w, height=render_h)
        img_reader = ImageReader(io.BytesIO(png_bytes))

        col_idx = pos % cols
        row_idx = pos // cols
        x = margin + col_idx * (img_w + col_gap)
        y = page_h - margin - (row_idx + 1) * img_h - row_idx * row_gap
        c.drawImage(img_reader, x, y, width=img_w, height=img_h, preserveAspectRatio=False)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

####################
### common text ####
####################

allowed_formats = "Allowed formats: csv (comma separated), tsv (tab separated), txt (tab separated), xlsx (Excel file)."


def get_feature_name_map():
    """Return a mapping metabolite_id -> feature name from ft_gnps, or None."""
    ft = st.session_state.get("ft_gnps", pd.DataFrame())
    if ft is None or ft.empty:
        return None
    candidates = ["metabolite_name", "name", "feature_name", "compound_name", "compound"]
    for c in candidates:
        if c in ft.columns:
            return ft[c].to_dict()
    return None

#########################
### useful functions ####
#########################


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
        return df
    except:
        return pd.DataFrame()


def string_overlap(string, options):
    for option in options:
        if option in string and "mzml" not in string:
            return True
    return False


def table_title(df, title, col=""):
    text = f"##### {title}\n{df.shape[0]} rows, {df.shape[1]} columns"
    if col:
        col.markdown(text)
        col.download_button(
            "Download Table",
            df.to_csv(sep="\t").encode("utf-8"),
            title.replace(" ", "-") + ".tsv",
        )
    else:
        st.markdown(text)
        st.download_button(
            "Download Table",
            df.to_csv(sep="\t").encode("utf-8"),
            title.replace(" ", "-") + ".tsv",
        )


patterns = [
    ["m/z", "mz", "mass over charge"],
    ["rt", "retention time", "retention-time", "retention_time"],
]


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
        # set metabolites column with index as default
        df["metabolite"] = df.index
        if len(column_names) == 2:
            df["metabolite"] = df[column_names[0]].round(5).astype(str)
            if column_names[1]:
                df["metabolite"] = (
                    df["metabolite"] + "@" + df[column_names[1]].round(2).astype(str)
                )
            if "row ID" in df.columns:
                df["metabolite"] = df["row ID"].astype(str) + "_" + df["metabolite"]
        df.set_index("metabolite", inplace=True)
    except:
        return df, "fail"
    return df, "success"


def compute_dominant_groups(metabolites, color_by, sample_filter_column=None, sample_filter_values=None):
    """For each metabolite, determine which group in metadata column color_by has the highest mean intensity."""
    import numpy as np
    data = st.session_state.data
    md = st.session_state.md

    if sample_filter_column is not None and sample_filter_values:
        md = md[md[sample_filter_column].isin(sample_filter_values)]

    valid_idx = md.index.intersection(data.index)
    md = md.loc[valid_idx]
    data = data.loc[valid_idx]

    groups = sorted(str(g) for g in md[color_by].dropna().unique())
    if not groups:
        return {}, []

    result = {}
    for met in metabolites:
        if met not in data.columns:
            continue
        best_group, best_mean = groups[0], -np.inf
        for g in groups:
            mask = md[color_by].astype(str) == g
            valid_idx = mask[mask].index.intersection(data.index)
            if len(valid_idx) == 0:
                continue
            m = data.loc[valid_idx, met].mean()
            if pd.notnull(m) and m > best_mean:
                best_mean, best_group = m, g
        result[met] = best_group
    return result, groups


def inside_levels(df):
    # get all the columns (equals all attributes) -> will be number of rows
    levels = []
    # types = []
    count = []
    for col in df.columns:
        # types.append(type(df[col][0]))
        levels.append(sorted(set(df[col].dropna())))
        tmp = df[col].value_counts()
        count.append([tmp[levels[-1][i]] for i in range(len(levels[-1]))])
    return pd.DataFrame(
        {"ATTRIBUTES": df.columns, "LEVELS": levels, "COUNT": count},
        index=range(1, len(levels) + 1),
    )


def download_plotly_figure(fig, col=None, filename=""):
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


