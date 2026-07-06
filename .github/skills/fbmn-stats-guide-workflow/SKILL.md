---
name: fbmn-stats-guide-workflow
description: 'Run the complete FBMN-STATS-GUIDE metabolomics workflow: data loading, cleanup, PCA, PCoA+PERMANOVA, clustering heatmap, random forest, parametric assumptions, and statistical test recommendation/execution (ANOVA+Tukey, RM-ANOVA, t-test Student/Welch, Kruskal+Dunn, Mann-Whitney, Wilcoxon signed-rank, Friedman). Use for advising users which test to run from their files.'
argument-hint: 'Provide feature table, metadata table, analysis goal, and whether samples are paired.'
user-invocable: true
disable-model-invocation: false
---

# FBMN-STATS Guide Workflow

## Purpose
Apply the same end-to-end workflow and requirements as the FBMN-STATS app to:
- validate user files,
- clean and normalize metabolomics data,
- run exploratory analyses,
- evaluate assumptions,
- recommend the correct statistical test,
- and report statistically sound results.

Ask the user for all required inputs and design details before running any analysis. If any required information is missing, stop and request it explicitly. Ask the user which inputs they would like to run analysis on and which outputs they would like to receive. Return the same output artifacts and tab-style sections used in the FBMN-STATS app.

## When To Use
- User asks for metabolomics statistical guidance from feature matrix + metadata.
- User needs help choosing among t-test, ANOVA, Kruskal-Wallis, Wilcoxon, Friedman, etc.
- User wants the full FBMN-STATS-GUIDE flow from ingestion to interpretation.

## Required Intake (Ask In This Order)
Always collect these before analysis. If missing, ask follow-up questions.

1. Data source:
- `Quantification + metadata files`
- `GNPS(2) FBMN task ID`
- `GNPS2 CMN task ID`
- `Example dataset`

2. File requirements:
- Quantification table format: `csv`, `tsv`, `txt`, or `xlsx`.
- Metadata table format: `csv`, `tsv`, `txt`, or `xlsx`.
- Metadata must include a sample identifier column (`filename` preferred).
- Feature table must have sample columns matching metadata sample IDs after cleanup (`.mzML/.mzXML` suffix removal allowed).
- Feature IDs should be unique. If no `metabolite` column exists, create IDs from m/z and RT when available.

3. Analysis design:
- Attribute (grouping variable) to analyze.
- Group levels to include.
- Are groups independent or paired/dependent?
- Number of groups to compare (2 or 3+).
- Alternative hypothesis for 2-group tests: `two-sided`, `greater`, `less`.

4. Statistical preferences:
- Multiple-testing correction method: `none`, `fdr_bh`, `sidak`, `bonf`, or `fdr_by`.
- For t-test: `auto` (Welch default), `Welch`, or `Student`.

## Method-Specific Questions (Must Ask Before Running)

1. PCA:
- Which metadata attribute should be used for filtering?
- Which categories/samples should be included?
- Which PCs should be plotted on X and Y axes?
- Which metadata column should color points?

2. PCoA + PERMANOVA:
- Which metadata attribute should be used for filtering?
- Which categories/samples should be included?
- Which distance metric should be used?
- Which metadata column should be used for color and PERMANOVA grouping?
- Confirm there are at least 2 categories and at least 2 samples per category for PERMANOVA.

3. Hierarchical Clustering + Heatmap:
- Which metadata attribute/categories/samples should be included?
- Any feature filters by name, m/z range, or RT range?
- Which heatmap palette should be used?

4. Random Forest:
- Which metadata attribute is the class label?
- Which categories should be included (must be >= 2)?
- Number of trees?
- Fixed random seed for reproducibility?

5. Parametric Assumptions:
- Which attribute and exactly two levels should be compared?
- Which equal-variance test should be used: Levene or Bartlett?

6. t-test:
- Exactly two group levels.
- Paired or independent?
- Test type: auto/Welch/Student.
- Alternative hypothesis.

7. ANOVA + Tukey:
- Attribute and at least 3 groups for ANOVA.
- Exactly two groups for Tukey post hoc pair view.

8. Repeated Measures ANOVA:
- Within-subject factor with at least 3 levels.
- Subject identifier column.
- Confirm repeated/paired structure.

9. Kruskal-Wallis + Dunn:
- Attribute and at least 3 groups for Kruskal-Wallis.
- Exactly two groups for Dunn post hoc pair view.

10. Mann-Whitney:
- Exactly two independent groups.
- Alternative hypothesis.

11. Wilcoxon Signed-Rank:
- Exactly two paired groups.
- Alternative hypothesis.
- Confirm paired alignment and effective equal paired count.

12. Friedman:
- At least 3 paired groups.
- Confirm paired alignment across conditions.
- If post hoc is requested, specify pairwise Wilcoxon comparisons.

## Workflow

### Step 1: Data Loading And Structural Validation
1. Load feature table and metadata.
2. Ensure metadata sample-ID column is set as index (`filename` preferred).
3. Clean sample IDs consistently:
- strip spaces,
- remove `.mzML`, `.mzXML`, and ` Peak area` suffix artifacts.
4. Remove non-sample columns from feature table unless used for feature ID generation.
5. Align feature columns and metadata rows by sample ID; drop unmatched entries with explicit warning.
6. Stop if feature IDs are non-unique.

### Step 2: Data Cleanup
1. Blank removal (optional but recommended):
- user chooses sample subset and blank subset,
- compute ratio `(mean(blank)+1)/(mean(sample)+1)`,
- remove features above cutoff (recommended cutoff range: `0.1` to `0.3`).
2. Missing value imputation (optional):
- replace zeros with random integers in `[1, LOD)` where `LOD` is the lowest non-zero value.
3. Normalization:
- `None`,
- `Center-Scaling`,
- `TIC/sample-centric`.
4. Confirm readiness before downstream statistics.



### Step 3: Exploratory Multivariate Analysis
1. PCA (unsupervised):
- require at least 2 samples,
- allow filtered categories/samples,
- select plotting axes from top components.
2. PCoA + PERMANOVA:
- choose distance metric,
- PCoA needs at least 2 samples,
- PERMANOVA requires at least 2 categories and at least 2 samples per category in the grouping variable.
3. Hierarchical clustering + heatmap:
- require at least 2 samples and at least 2 features after filters,
- cluster with Euclidean distance and complete linkage.

### Step 4: Random Forest (Supervised)
1. Choose classification attribute and include at least 2 categories.
2. Set number of trees.
3. Return OOB error trend, feature importance, classification report, and confusion matrices.

### Step 5: Parametric Assumptions Evaluation
MAKE SURE TO DO THIS STEP BEFORE CHOOSING A TEST. This step is required for all tests!

1. Select one attribute and exactly 2 levels for assumption diagnostics.
2. Normality check: Shapiro-Wilk across features.
3. Variance check:
- Levene when normality is questionable,
- Bartlett when normality is reasonably satisfied.
4. Summarize proportion of features with p-values > 0.05.

### Step 6: Test Recommendation Logic (Core Decision Tree)
Use this branching exactly:

1. Determine pairing:
- `paired/dependent`
- `independent`

2. Determine number of groups:
- `2 groups`
- `3+ groups`

3. Use assumptions:
- Normality (from Shapiro-Wilk)
- Equal variance (Levene/Bartlett)

4. Recommend test:
- 2 groups, independent, approximately normal, equal variance: `Student t-test`
- 2 groups, independent, approximately normal, unequal variance: `Welch t-test`
- 2 groups, independent, non-normal: `Mann-Whitney U`
- 2 groups, paired, approximately normal: `Paired t-test`
- 2 groups, paired, non-normal: `Wilcoxon Signed-Rank`
- 3+ groups, independent, approximately normal: `One-way ANOVA`, then `Tukey` for pairwise post hoc
- 3+ groups, independent, non-normal: `Kruskal-Wallis`, then `Dunn` for pairwise post hoc
- 3+ groups, paired, approximately normal: `Repeated Measures ANOVA`
- 3+ groups, paired, non-normal: `Friedman`

### Step 7: Per-Test Constraints And Guards
Apply these checks before running each test:
- ANOVA / Kruskal-Wallis: minimum 3 groups.
- Tukey / Dunn: exactly 2 selected groups for post hoc pair visualization.
- t-test / Mann-Whitney / Wilcoxon: exactly 2 groups.
- Repeated Measures ANOVA: subject ID column required; each subject should have one observation per condition.
- Wilcoxon/Friedman: paired structure required; groups must be alignable by pairing and effectively same paired count.

### Step 8: Results And Interpretation
Always provide:
1. Test statistic and p-value.
2. Corrected p-value using selected correction method.
3. Significant vs non-significant counts (`p-corrected < 0.05`).
4. Ranked feature table by corrected p-value.
5. Recommended next action:
- run post hoc test,
- inspect boxplots/volcano/test-stat plots,
- refine groups or assumptions.

## Required Outputs (Matching FBMN-STATS-GUIDE)
The skill must return the same output artifacts and tab-style sections used in the app.

1. Data Preparation page outputs:
- Tables/previews for uploaded or loaded sources: quantification, metadata, annotation (optional), node-pair (optional), and combined final table (`FeatureMatrix-scaled-centered`).
- Data Cleanup tabs: `**Blank Removal**`, `**Imputation**`, `**Normalization**`, `📊 **Summary**`.
- Summary plots with the same intent as app figures:
	- `📊 Feature intensity frequency` (`feature-intensity-frequency`)
	- `📊 Missing values per feature` (`missing-values`)

2. PCA page outputs:
- Tabs exactly: `📈 PCA Scores Plot`, `📊 Explained variance`, `📁 Data`.
- Outputs:
	- PCA scatter (`principal-component-analysis`)
	- Variance figure (`pca-variance`)
	- Principal components table (`principal-components`)

3. PERMANOVA and PCoA page outputs:
- If PERMANOVA is valid, tabs exactly: `📁 PERMANOVA statistics`, `📈 Principal Coordinate Analysis`, `📊 Explained variance`, `📁 Data`.
- If PERMANOVA is not valid, tabs exactly: `📈 Principal Coordinate Analysis`, `📊 Explained variance`, `📁 Data`.
- Outputs:
	- PERMANOVA table (`PERMANOVA-statistics`) when available
	- PCoA scatter (`principal-coordinate-analysis`)
	- PCoA variance figure (`pcoa-variance`)
	- Principal coordinates table (`principal-coordinates`)

4. Hierarchical Clustering and Heatmap page outputs:
- Tabs exactly: `🧬 Clustered Heatmap`, `📁 Heatmap Data`.
- Outputs:
	- Clustered heatmap (`clustermap`)
	- Heatmap data table (`heatmap-data`)

5. Random Forest page outputs:
- Main tabs exactly: `📈 Analyze optimum number of trees`, `📁 Feature ranked by importance`, `📋 Classification Report`, `🔍 Confusion Matrix`.
- Feature-importance subtabs exactly: `📁 Table`, `📊 Plot`.
- Outputs:
	- OOB error figure (`oob-error`)
	- Feature importance figure (`feature-importance`) and table
	- Classification report table and metrics
	- Train/test confusion matrices and accuracies

6. Parametric Assumptions Evaluation page outputs:
- Tabs exactly: `📊 Normality Check`, `📊 Equal Variance Check`.
- Outputs:
	- Normality figure (`test-normal-distribution`)
	- Equal-variance figure based on choice:
		- Levene (`test-equal-variance`)
		- Bartlett (`test-equal-variance-bartlett`)

7. One-way ANOVA and Tukey page outputs:
- Top tabs exactly: `ANOVA` and then `Tukey's` after ANOVA exists.
- ANOVA subtabs exactly: `📈 ANOVA: plot`, `📁 ANOVA: result table`, `📊 ANOVA: metabolites (boxplots)`.
- Tukey subtabs exactly: `📈 Tukey's: plots`, `📁 Tukey's: result table`.
- Outputs:
	- ANOVA figure (`anova`)
	- ANOVA metabolite boxplot figure (`anova-<metabolite>`)
	- Tukey test-stat figure (`tukeys-teststat`)
	- Tukey volcano figure (`tukeys-volcano`)
	- Result tables for both ANOVA and Tukey
	- PDF download flow for ANOVA boxplots

8. Repeated Measures ANOVA page outputs:
- Tabs exactly: `📈 RM ANOVA: plot`, `📁 RM ANOVA: result table`, `📊 RM ANOVA: metabolites (boxplots)`.
- Outputs:
	- RM-ANOVA figure (`rm_anova`)
	- RM-ANOVA metabolite boxplot (`rm_anova-<metabolite>`)
	- RM-ANOVA result table
	- PDF download flow for RM-ANOVA boxplots

9. T-test page outputs:
- Tabs exactly: `📈 Feature significance`, `📈 Volcano plot`, `📊 Single metabolite plots`, `📁 Data`.
- Outputs:
	- Feature significance figure (`t-test`)
	- Volcano figure (`ttest-volcano`)
	- Boxplot figure (`ttest-boxplot-<metabolite>`)
	- T-test result table
	- PDF download flow for t-test boxplots

10. Kruskal-Wallis and Dunn page outputs:
- Top tabs exactly: `Kruskal-Wallis` and then `Dunn's` after Kruskal results exist.
- Kruskal subtabs exactly: `📈 KW: plot`, `📁 KW: result table`, `📊 KW: metabolites (boxplots)`.
- Dunn subtabs exactly: `📈 Dunn's: plots`, `📁 Dunn's: result table`.
- Outputs:
	- Kruskal figure (`kruskal`)
	- Kruskal boxplot (`kruskal-<metabolite>`)
	- Dunn test-stat figure (`dunn-teststat`)
	- Dunn volcano figure (`dunn-volcano`)
	- Result tables for both KW and Dunn
	- PDF download flow for KW boxplots

11. Mann-Whitney page outputs:
- Tabs exactly: `📈 Feature significance`, `📊 Single metabolite plots`, `📁 Data`.
- Outputs:
	- Feature significance figure (`mwu`)
	- Boxplot figure (`mwu-boxplot-<metabolite>`)
	- Result table
	- PDF download flow for MWU boxplots

12. Wilcoxon Signed-Rank page outputs:
- Tabs exactly: `📈 Feature significance`, `📊 Single metabolite plots`, `📁 Data`.
- Outputs:
	- Feature significance figure (`wilcoxon-feature-significance`)
	- Boxplot figure (`wilcoxon-boxplot-<metabolite>`)
	- Result table
	- PDF download flow for Wilcoxon boxplots

13. Friedman page outputs:
- Tabs exactly:
	- `📈 Friedman: plot`
	- `📁 Friedman: result table`
	- `📊 Friedman: metabolites (boxplots)`
- Outputs:
	- Friedman figure (`friedman`)
	- Friedman boxplot (`friedman-<metabolite>`)
	- Friedman result table
	- PDF download flow for Friedman boxplots

14. Cross-module output rules:
- Always return both plot(s) and table(s) when both exist in the app.
- Always report significant count, insignificant count, and total count where significance columns exist.
- Always apply selected p-value correction in final significance calls.
- Always state selected attribute, selected groups/conditions, and pairing mode used.

## Quality Checks Before Completion
Confirm all are true:
- Input files are validated and aligned.
- Cleanup decisions were explicit (blank removal/imputation/normalization).
- Pairing and group-count logic were explicitly checked.
- Test choice was justified with assumptions and design.
- Multiple-testing correction was applied and reported.
- Outputs include both statistical and practical interpretation.
- Module outputs match FBMN-STATS-GUIde output artifacts (plots, tables, and exportable views).

## Failure Handling
If a requirement fails, stop and request exactly what is missing:
- missing metadata sample-ID column,
- fewer than required groups,
- fewer than 2 samples in a selected category,
- invalid paired design,
- no features remaining after cleanup/filtering.

Do not guess missing design details.

## Example Prompts
- `Run FBMN-STATS workflow on these files and tell me if I should use ANOVA or Kruskal-Wallis.`
- `I have paired before/after samples. Use the FBMN workflow and recommend the correct 2-group test.`
- `Use the full FBMN pipeline from cleanup to PERMANOVA and then select the right univariate tests.`

## General Code Templates (All Tests And Outputs)
Use these as implementation patterns. Keep names aligned with the output artifact names listed above.

```python
# General imports
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.stats import mannwhitneyu, wilcoxon, friedmanchisquare, kruskal, levene, bartlett, shapiro


# -------------------------------
# 1) DATA LOADING + VALIDATION
# -------------------------------
def load_tables(feature_path, metadata_path):
	ft = pd.read_csv(feature_path) if feature_path.endswith(".csv") else pd.read_table(feature_path)
	md = pd.read_csv(metadata_path) if metadata_path.endswith(".csv") else pd.read_table(metadata_path)
	if "filename" in md.columns:
		md = md.set_index("filename")
	return ft, md

def clean_ids(ft, md):
	md.index = md.index.astype(str).str.strip().str.replace(".mzML", "", regex=False).str.replace(".mzXML", "", regex=False)
	ft.columns = [str(c).replace(" Peak area", "").replace(".mzML", "").replace(".mzXML", "").strip() for c in ft.columns]
	return ft, md

def align_tables(ft, md, metabolite_col="metabolite"):
	if metabolite_col in ft.columns:
		ft = ft.set_index(metabolite_col)
	common = [c for c in ft.columns if c in md.index]
	ft = ft[common]
	md = md.loc[common]
	if not ft.index.is_unique:
		raise ValueError("Feature IDs must be unique.")
	return ft, md


# -------------------------------
# 2) CLEANUP OUTPUTS
# -------------------------------
def remove_blank_features(ft, md, sample_col, sample_vals, blank_col, blank_vals, cutoff=0.3):
	sample_idx = md[md[sample_col].isin(sample_vals)].index
	blank_idx = md[md[blank_col].isin(blank_vals)].index
	s = ft[sample_idx].mean(axis=1)
	b = ft[blank_idx].mean(axis=1)
	keep = ((b + 1) / (s + 1)) < cutoff
	return ft.loc[keep], int((~keep).sum())

def impute_zeros(ft):
	lod = ft.replace(0, np.nan).min().min()
	if pd.isna(lod) or lod <= 1:
		return ft
	out = ft.copy()
	mask = out == 0
	out[mask] = np.random.randint(1, int(lod), size=mask.sum().sum())
	return out

def normalize(ft_t, method="None"):
	if method == "Center-Scaling":
		return pd.DataFrame(StandardScaler().fit_transform(ft_t), index=ft_t.index, columns=ft_t.columns)
	if method == "Total Ion Current (TIC) or sample-centric normalization":
		return ft_t.div(ft_t.sum(axis=1), axis=0)
	return ft_t


# -------------------------------
# 3) PCA OUTPUTS
# -------------------------------
def run_pca(data_samples_x_features, n_components=10):
	n_components = min(n_components, data_samples_x_features.shape[0], data_samples_x_features.shape[1])
	model = PCA(n_components=n_components)
	scores = model.fit_transform(data_samples_x_features)
	pca_df = pd.DataFrame(scores, index=data_samples_x_features.index, columns=[f"PC{i+1}" for i in range(n_components)])
	var_df = pd.DataFrame({"component": pca_df.columns, "explained_variance_ratio": model.explained_variance_ratio_})
	return pca_df, var_df


# -------------------------------
# 4) PCoA + PERMANOVA OUTPUTS
# -------------------------------
def pcoa_from_distance(data_samples_x_features, metric="euclidean"):
	d = squareform(pdist(data_samples_x_features.values, metric=metric))
	d2 = d ** 2
	n = d2.shape[0]
	J = np.eye(n) - np.ones((n, n)) / n
	B = -0.5 * J @ d2 @ J
	eigvals, eigvecs = np.linalg.eigh(B)
	idx = np.argsort(eigvals)[::-1]
	eigvals = eigvals[idx]
	eigvecs = eigvecs[:, idx]
	pos = eigvals > 0
	coords = eigvecs[:, pos] * np.sqrt(eigvals[pos])
	cols = [f"PC{i+1}" for i in range(coords.shape[1])]
	pcoa_df = pd.DataFrame(coords, index=data_samples_x_features.index, columns=cols)
	var = eigvals[pos] / eigvals[pos].sum() if pos.any() else np.array([])
	var_df = pd.DataFrame({"component": cols, "explained_variance_ratio": var})
	return pcoa_df, var_df, d

def run_permanova(distance_matrix, groups, permutations=999):
	# General template: replace with your project PERMANOVA utility if available.
	# Return table with pseudo-F, p-value, R2.
	return pd.DataFrame([{"pseudo-F": np.nan, "p-value": np.nan, "R2": np.nan, "permutations": permutations}])


# -------------------------------
# 5) HCA + HEATMAP OUTPUTS
# -------------------------------
def run_hca(data_samples_x_features, metric="euclidean", method="complete"):
	Z = linkage(data_samples_x_features.values, method=method, metric=metric)
	# Return linkage matrix for dendrogram + filtered matrix for heatmap table
	return Z, data_samples_x_features.copy()


# -------------------------------
# 6) RANDOM FOREST OUTPUTS
# -------------------------------
def run_rf(data_samples_x_features, labels, n_trees=100, random_state=123):
	rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, bootstrap=True, random_state=random_state)
	rf.fit(data_samples_x_features, labels)
	pred = rf.predict(data_samples_x_features)
	out = {
		"oob_error": 1.0 - getattr(rf, "oob_score_", np.nan),
		"feature_importance": pd.DataFrame({"feature": data_samples_x_features.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False),
		"classification_report": pd.DataFrame(classification_report(labels, pred, output_dict=True)).T,
		"confusion_matrix": pd.DataFrame(confusion_matrix(labels, pred)),
		"accuracy": float(accuracy_score(labels, pred)),
	}
	return out


# -------------------------------
# 7) PARAMETRIC ASSUMPTIONS OUTPUTS
# -------------------------------
def featurewise_shapiro(data, group_a_idx, group_b_idx):
	rows = []
	for met in data.columns:
		a = data.loc[group_a_idx, met].dropna().astype(float)
		b = data.loc[group_b_idx, met].dropna().astype(float)
		if len(a) >= 3:
			rows.append((met, "A", shapiro(a).pvalue))
		if len(b) >= 3:
			rows.append((met, "B", shapiro(b).pvalue))
	return pd.DataFrame(rows, columns=["metabolite", "group", "p"])

def featurewise_variance_test(data, group_a_idx, group_b_idx, test_name="levene"):
	rows = []
	for met in data.columns:
		a = data.loc[group_a_idx, met].dropna().astype(float)
		b = data.loc[group_b_idx, met].dropna().astype(float)
		if len(a) >= 2 and len(b) >= 2:
			p = levene(a, b).pvalue if test_name == "levene" else bartlett(a, b).pvalue
			rows.append((met, p))
	return pd.DataFrame(rows, columns=["metabolite", "p"])


# -------------------------------
# 8) UNIVARIATE TESTS + OUTPUT TABLES
# -------------------------------
def apply_p_correction(df, p_col="p", method="fdr_bh"):
	if df.empty:
		return df
	pvals = pd.to_numeric(df[p_col], errors="coerce").fillna(1.0).values
	_, p_corr = pg.multicomp(pvals, method=method)
	out = df.copy()
	out["p-corrected"] = p_corr
	out["significant"] = out["p-corrected"] < 0.05
	return out.sort_values("p-corrected")

def run_ttest_all(data, md, attr, groups, paired=False, correction="fdr_bh", alternative="two-sided", ttype="auto"):
	g1, g2 = groups
	i1 = md.index[md[attr] == g1]
	i2 = md.index[md[attr] == g2]
	rows = []
	for met in data.columns:
		try:
			r = pg.ttest(data.loc[i1, met], data.loc[i2, met], paired=paired, alternative=alternative, correction=(None if ttype == "auto" else (ttype == "Welch's")))
			rows.append({"metabolite": met, "T": float(r["T"].iloc[0]), "p": float(r["p-val"].iloc[0])})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_anova_all(data, md, attr, groups, correction="fdr_bh"):
	idx = md.index[md[attr].isin(groups)]
	rows = []
	joined = data.loc[idx].join(md[[attr]])
	for met in data.columns:
		try:
			r = pg.anova(data=joined[[met, attr]].dropna(), dv=met, between=attr, detailed=True)
			rows.append({"metabolite": met, "F": float(r["F"].iloc[0]), "p": float(r["p-unc"].iloc[0])})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_tukey_for_pair(data, md, attr, group_a, group_b, significant_mets, correction="fdr_bh"):
	idx = md.index[md[attr].isin([group_a, group_b])]
	rows = []
	joined = data.loc[idx].join(md[[attr]])
	for met in significant_mets:
		try:
			t = pg.pairwise_tukey(joined[[met, attr]].dropna(), dv=met, between=attr)
			t = t[(t["A"].astype(str).isin([str(group_a), str(group_b)])) & (t["B"].astype(str).isin([str(group_a), str(group_b)]))]
			if not t.empty:
				rows.append({"metabolite": met, "diff": float(t["diff"].iloc[0]), "p": float(t["p-tukey"].iloc[0])})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_rm_anova_all(data, md, within_attr, subject_col, groups, correction="fdr_bh"):
	sub_md = md[md[within_attr].isin(groups)].copy()
	rows = []
	for met in data.columns:
		try:
			tmp = pd.DataFrame({
				"dv": data.loc[sub_md.index, met].values,
				within_attr: sub_md[within_attr].values,
				subject_col: sub_md[subject_col].values,
			}).dropna()
			r = pg.rm_anova(data=tmp, dv="dv", within=within_attr, subject=subject_col, detailed=True)
			rows.append({"metabolite": met, "F": float(r["F"].iloc[0]), "p": float(r["p-unc"].iloc[0])})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_kruskal_all(data, md, attr, groups, correction="fdr_bh"):
	idx = md.index[md[attr].isin(groups)]
	rows = []
	for met in data.columns:
		try:
			arrays = [data.loc[idx[md.loc[idx, attr] == g], met].dropna().values for g in groups]
			arrays = [a for a in arrays if len(a) > 0]
			if len(arrays) >= 3:
				stat, p = kruskal(*arrays)
				rows.append({"metabolite": met, "K": float(stat), "p": float(p)})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_dunn_for_pair(data, md, attr, group_a, group_b, significant_mets, correction="fdr_bh"):
	# General placeholder: your project can use a dedicated Dunn implementation.
	rows = []
	for met in significant_mets:
		rows.append({"metabolite": met, "rank_sum_diff": np.nan, "p": np.nan})
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_mwu_all(data, md, attr, groups, correction="fdr_bh", alternative="two-sided"):
	g1, g2 = groups
	i1 = md.index[md[attr] == g1]
	i2 = md.index[md[attr] == g2]
	rows = []
	for met in data.columns:
		try:
			stat, p = mannwhitneyu(data.loc[i1, met].dropna(), data.loc[i2, met].dropna(), alternative=alternative)
			rows.append({"metabolite": met, "U": float(stat), "p": float(p)})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_wilcoxon_all(data, md, attr, groups, correction="fdr_bh", alternative="two-sided"):
	g1, g2 = groups
	i1 = list(md.index[md[attr] == g1])
	i2 = list(md.index[md[attr] == g2])
	n = min(len(i1), len(i2))
	i1, i2 = i1[:n], i2[:n]
	rows = []
	for met in data.columns:
		try:
			stat, p = wilcoxon(data.loc[i1, met].values, data.loc[i2, met].values, alternative=alternative)
			rows.append({"metabolite": met, "W": float(stat), "p": float(p)})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)

def run_friedman_all(data, md, attr, groups, correction="fdr_bh"):
	idx_map = {g: list(md.index[md[attr] == g]) for g in groups}
	n = min(len(v) for v in idx_map.values())
	idx_map = {k: v[:n] for k, v in idx_map.items()}
	rows = []
	for met in data.columns:
		try:
			arrays = [data.loc[idx_map[g], met].values for g in groups]
			stat, p = friedmanchisquare(*arrays)
			rows.append({"metabolite": met, "statistic": float(stat), "p": float(p)})
		except Exception:
			continue
	return apply_p_correction(pd.DataFrame(rows), "p", correction)


# -------------------------------
# 9) OUTPUT HELPERS (PLOTS/TABLES/PDF)
# -------------------------------
def summarize_significance(df, sig_col="significant"):
	if df.empty or sig_col not in df.columns:
		return {"significant": 0, "insignificant": 0, "total": len(df)}
	s = int(df[sig_col].sum())
	t = len(df)
	return {"significant": s, "insignificant": t - s, "total": t}

def select_top_metabolites_for_boxplots(df, mode="Top N significant", n=10):
	if df.empty:
		return []
	pcol = "p-corrected" if "p-corrected" in df.columns else "p"
	if mode == "Single metabolite":
		return [df.iloc[0]["metabolite"]] if "metabolite" in df.columns else []
	want_sig = mode == "Top N significant"
	if "significant" in df.columns:
		pool = df[df["significant"] == want_sig]
	else:
		pool = df
	return list(pool.sort_values(pcol).head(n)["metabolite"]) if "metabolite" in pool.columns else []
```
