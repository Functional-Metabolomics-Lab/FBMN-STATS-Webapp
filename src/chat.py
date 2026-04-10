import os
import requests
import json
import base64
import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# Load Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLEGEMINIAPI")


def get_current_page_name():
	"""Try to figure out which page is active right now."""
	if "current_page" in st.session_state:
		return st.session_state["current_page"]
	if "page_name" in st.session_state:
		return st.session_state["page_name"]
	return "Current analysis page"


def build_analysis_context_summary():
	"""Create a short text summary about the app state for Gemini."""

	parts = []
	page_name = get_current_page_name()
	parts.append("Current page: %s" % page_name)

	# Include dataset dimensions
	data = st.session_state.get("data")
	if isinstance(data, pd.DataFrame) and not data.empty:
		parts.append("Dataset loaded: %d samples × %d features" % (data.shape[0], data.shape[1]))

	md = st.session_state.get("md")
	if isinstance(md, pd.DataFrame) and not md.empty:
		parts.append("Metadata: %d samples, attributes: %s" % (md.shape[0], list(md.columns)))

	# Include current parameter settings relevant to the active page
	param_keys = [
		"pca_attribute", "pcoa_attribute", "hca_committed_attribute",
		"rf_attribute", "anova_attribute", "kruskal_attribute",
		"ttest_attribute", "mwu_attribute", "test_attribute",
		"pcoa_distance_matrix", "rf_n_trees",
		"pca_committed_categories", "pca_committed_samples",
		"pcoa_committed_categories", "pcoa_committed_samples",
		"hca_committed_categories", "hca_committed_samples",
		"rf_categories", "anova_groups", "kruskal_groups",
		"ttest_options", "mwu_options",
	]
	param_summary = []
	for key in param_keys:
		val = st.session_state.get(key)
		if val is not None:
			param_summary.append("%s: %s" % (key, val))
	if param_summary:
		parts.append("Current parameter settings:\n" + "\n".join("- " + p for p in param_summary))

	# Look for important dataframes and include snippets
	important_df_keys = ["df_anova", "df_ttest", "df_kruskal", "df_important_features", "df_tukey", "df_dunn", "permanova"]
	for key in important_df_keys:
		df = st.session_state.get(key)
		if isinstance(df, pd.DataFrame) and not df.empty:
			# Include top 8 rows of results, sorted by p-value if applicable
			try:
				if "p-corrected" in df.columns:
					snippet_df = df.sort_values("p-corrected").head(8)
				elif "p-value" in df.columns:
					snippet_df = df.sort_values("p-value").head(8)
				else:
					snippet_df = df.head(8)
				parts.append("Dataframe '%s' (top results):\n%s" % (key, snippet_df.to_string()))
			except Exception:
				parts.append("Dataframe '%s': shape=%s" % (key, str(df.shape)))

	# Summarize other dataframes
	df_summaries = []
	for key, value in st.session_state.items():
		if isinstance(value, pd.DataFrame) and not value.empty and key not in important_df_keys:
			cols_preview = list(value.columns)[:10]
			info = "%s: shape=%s, columns=%s" % (key, str(value.shape), cols_preview)
			df_summaries.append(info)

	if df_summaries:
		parts.append("Other dataframes available:\n" + "\n".join("- " + s for s in df_summaries))

	# Check for specific analysis results
	analysis_results = st.session_state.get("analysis_results")
	if analysis_results:
		parts.append("Latest Analysis Results (%s):" % analysis_results.get("type", "Unknown"))
		results_str = json.dumps(analysis_results, indent=2, default=str)
		parts.append(results_str)

	fig = st.session_state.get("last_figure")
	if fig is not None:
		try:
			x_title = getattr(fig.layout.xaxis.title, "text", "") if hasattr(fig.layout, "xaxis") else ""
			y_title = getattr(fig.layout.yaxis.title, "text", "") if hasattr(fig.layout, "yaxis") else ""
		except Exception:
			x_title, y_title = "", ""

		parts.append(
			"A Plotly figure is available with %d traces. X-axis: '%s', Y-axis: '%s'."
			% (len(getattr(fig, "data", [])), x_title, y_title)
		)

	# Include page-specific figures in text context
	page = get_current_page_name()
	page_fig_keys = _PAGE_FIGURE_KEYS.get(page, [])
	for key in page_fig_keys:
		page_fig = st.session_state.get(key)
		if page_fig is not None:
			try:
				title = getattr(page_fig.layout.title, "text", key)
			except Exception:
				title = key
			parts.append("Figure '%s' with %d traces." % (title, len(getattr(page_fig, "data", []))))

	return "\n\n".join(parts)

@st.cache_data
def get_app_summary():
	"""Read the high-level app description from 0_Home.txt."""
	return get_prompt_content("0_Home.txt")

def get_prompt_content(filename):
	"""Read content from a specific prompt file in assets/prompts."""
	try:
		path = os.path.join("assets", "prompts", filename)
		if os.path.exists(path):
			with open(path, "r", encoding="utf-8") as f:
				text = f.read()
				# Remove context tags
				text = text.replace("[PAGE_CONTEXT]", "").replace("[/PAGE_CONTEXT]", "").strip()
				return text
	except Exception:
		pass
	return ""

def get_current_page_prompt():
	"""Get the specific prompt content for the current page."""
	page_map = {
		"Home": "0_Home.txt",
		"Data Preparation": "1_Data_Preparation.txt",
		"PCA": "2_PCA.txt",
		"PERMANOVA & PCoA": "3_PERMANOVA.txt",
		"Hierarchical Clustering & Heatmap": "4_Clustering_Heatmap.txt",
		"Random Forest": "5_Random_Forest.txt",
		"Parametric Assumptions Evaluation": "6_PAE.txt",
		"ANOVA & Tukey's": "7_ANOVA_Tukeys.txt",
		"Kruskal-Wallis & Dunn's": "8_KW_Dunns.txt",
		"T-test": "9_T-test.txt",
		"Mann-Whitney U": "10_MW.txt"
	}
	
	current_page = get_current_page_name()
	filename = page_map.get(current_page)
	if filename:
		return get_prompt_content(filename)
	return ""

_PAGE_FIGURE_KEYS = {
	"Parametric Assumptions Evaluation": ["pae_normality_fig", "pae_variance_fig"],
	"PCA": ["page_figs_pca_scores", "page_figs_pca_scree"],
	"PERMANOVA & PCoA": ["page_figs_pcoa_scatter", "page_figs_pcoa_variance"],
	"Random Forest": ["page_figs_rf_oob"],
	"ANOVA & Tukey's": ["page_figs_anova_plot", "page_figs_anova_boxplot", "page_figs_tukey_teststat", "page_figs_tukey_volcano"],
	"T-test": ["page_figs_ttest_sig", "page_figs_ttest_volcano", "page_figs_ttest_boxplot"],
	"Kruskal-Wallis & Dunn's": ["page_figs_kw_plot", "page_figs_kw_boxplot", "page_figs_dunn_teststat", "page_figs_dunn_volcano"],
	"Mann-Whitney U": ["page_figs_mwu_sig", "page_figs_mwu_boxplot"],
	"Wilcoxon Signed-Rank": ["page_figs_wilcoxon_sig", "page_figs_wilcoxon_boxplot"],
	"Friedman": ["page_figs_friedman_plot", "page_figs_friedman_boxplot"],
}

def _figure_to_base64(fig):
	"""Convert a single Plotly figure to base64 PNG string."""
	if fig is None:
		return None
	try:
		import plotly.io as pio
		img_bytes = pio.to_image(fig, format="png", width=1000, height=800)
		return base64.b64encode(img_bytes).decode("utf-8")
	except Exception as e:
		print("Error converting figure to image: %s" % e)
		return None

def get_last_figure_base64():
	"""Convert the last stored Plotly figure in session state to base64 string."""
	return _figure_to_base64(st.session_state.get("last_figure"))

def get_all_figures_base64():
	"""Collect base64 images for all figures relevant to the current page."""
	images = []
	page = get_current_page_name()
	for key in _PAGE_FIGURE_KEYS.get(page, []):
		b64 = _figure_to_base64(st.session_state.get(key))
		if b64:
			images.append(b64)
	if not images:
		b64 = get_last_figure_base64()
		if b64:
			images.append(b64)
	return images

def call_gemini_with_context(user_message):
	"""Call Gemini with a prompt that includes app context and the user question."""

	if not GEMINI_API_KEY:
		return "Gemini API key is not configured. Please set GOOGLE_API_KEY in your .env file."

	context = build_analysis_context_summary()
	app_summary = get_app_summary()
	page_prompt = get_current_page_prompt()

	system_instructions = (
		"You are a specialized assistant embedded in a metabolomics statistical analysis web app called 'FBMN-STATS-GUIde'. "
		"This app allows for downstream statistical analysis for data within the GNPS2 workflow. "
		"Your ONLY role is to help users with analyses, results, and decisions WITHIN this app. "
		"You must REFUSE to answer any question that is unrelated to metabolomics, statistics, or the analysis pages in this app. "
		"If a user asks about anything outside this scope (e.g., general coding, cooking, trivia, other software), "
		"politely decline and redirect them to ask about their metabolomics data or one of the app's analysis pages.\n\n"

		"## Your Core Responsibilities\n\n"

		"### 1. Parameter Recommendations\n"
		"Proactively recommend which parameters to set on the current page based on the user's data and goals. "
		"Use the page context (provided below) to explain what each parameter does and suggest sensible values. "
		"For example: recommend a distance metric for PCoA (e.g., Bray-Curtis for relative abundance data), "
		"advise on the number of trees for Random Forest, explain which attribute to group by, "
		"or guide the user on advanced filtering to include or exclude specific samples or categories.\n\n"

		"### 2. Interpreting Results (Tables and Visualizations)\n"
		"When result tables or figures are available in the app context, interpret them clearly for the user. "
		"For tables: explain what key columns mean (p-value, test statistic, R², effect size, FDR correction, etc.), "
		"highlight the most significant features, and describe the overall pattern or story the data tells. "
		"For figures and plots: describe what the visualization shows, explain clustering patterns, "
		"group separation, variance explained, or any notable trends visible in the chart. "
		"Always ground your interpretation in the actual values from the data context provided to you.\n\n"

		"### 3. Next Page / Analysis Recommendations\n"
		"Based on the current results and page, actively suggest which page or analysis the user should run next. "
		"Follow these evidence-based guidelines:\n"
		"- If PCA or PCoA shows clear group separation → suggest PERMANOVA to confirm statistical significance.\n"
		"- If parametric assumption tests (Shapiro-Wilk, Levene) show many p < 0.05 → recommend Kruskal-Wallis or Mann-Whitney instead of ANOVA or t-test.\n"
		"- If parametric assumptions hold (p > 0.05) → recommend ANOVA (3+ groups) or t-test (2 groups).\n"
		"- If ANOVA or Kruskal-Wallis is significant → suggest Tukey's or Dunn's post-hoc test for pairwise comparisons.\n"
		"- If the user wants feature-level discriminators → suggest Random Forest.\n"
		"- If the user wants to visualize global patterns → suggest Hierarchical Clustering & Heatmap.\n"
		"Use the 'Next potential pages' sections in the page context to guide recommendations.\n\n"

		"### 4. Workflow and Next Step Guidance\n"
		"Guide the user through the full analytical workflow. Explain what each result means for their next decision. "
		"If results are inconclusive, suggest parameter changes or alternative approaches. "
		"Refer to previous results from the chat history to maintain continuity across the analysis session.\n\n"

		"### 5. Biological Interpretation\n"
		"Always provide a biological interpretation in the context of metabolomics whenever statistical results are available. "
		"Examples:\n"
		"- A significant PERMANOVA result means the groups have statistically distinct metabolic profiles.\n"
		"- High feature importance in Random Forest means those metabolites are key discriminators between conditions.\n"
		"- A significant t-test or Mann-Whitney result on a metabolite means its abundance differs meaningfully between the two groups.\n"
		"- Tight clusters in PCA or HCA suggest high metabolic similarity within groups.\n"
		"Keep biological interpretations grounded in the data context provided — do not speculate beyond what the data shows.\n\n"

		"## Important Rules\n"
		"- ONLY answer questions about this app, its pages, metabolomics data, or the statistical analyses being performed.\n"
		"- If a question is unrelated to the app or metabolomics, respond: 'I can only help with analyses within the Statistics for Metabolomics app. Please ask about your data or one of the analysis pages.'\n"
		"- Always base answers on the actual data context provided — never fabricate values or invent results.\n"
		"- If results or data are not yet available, say so clearly and tell the user how to generate them.\n"
		"- Be concise but thorough. Use bullet points and section headers for clarity where helpful.\n"
	)

	if app_summary:
		system_instructions += "\n\n---\n### General App Context:\n" + app_summary
	
	if page_prompt:
		system_instructions += "\n\n---\n### Current Page Context — %s:\n" % get_current_page_name() + page_prompt

	system_instructions += (
		"\n\n---\n"
		"Use the data context (dataframes, parameter settings, and figures) provided in the user message to answer accurately. "
		"If specific data is not available in the context, say so clearly rather than guessing."
	)

	# Build multi-turn conversation contents
	contents = []
	history = st.session_state.get("chat_history", [])
	
	img_b64_list = get_all_figures_base64()

	for i, (role, content) in enumerate(history):
		# For the current last user message, append the context information
		if i == len(history) - 1 and role == "user":
			text_with_context = "Context from the app:\n%s\n\nUser question: %s" % (context, content)
			parts = [{"text": text_with_context}]
			for img_b64 in img_b64_list:
				parts.append({
					"inline_data": {
						"mime_type": "image/png",
						"data": img_b64
					}
				})
			contents.append({
				"role": "user",
				"parts": parts
			})
		else:
			contents.append({
				"role": "user" if role == "user" else "model",
				"parts": [{"text": content}]
			})
	
	# Fallback if history is empty
	if not contents:
		parts = [{"text": "Context:\n%s\n\nQuestion: %s" % (context, user_message)}]
		for img_b64 in img_b64_list:
			parts.append({
				"inline_data": {
					"mime_type": "image/png",
					"data": img_b64
				}
			})
		contents.append({
			"role": "user",
			"parts": parts
		})

	try:
		# Use the latest flash alias to ensure compatibility in 2026
		url = (
			"https://generativelanguage.googleapis.com/v1beta/models/"
			"gemini-flash-latest:generateContent"
		)
		headers = {"Content-Type": "application/json"}
		payload = {
			"contents": contents,
			"system_instruction": {
				"parts": [{"text": system_instructions}]
			}
		}

		resp = requests.post(
			url,
			headers=headers,
			params={"key": GEMINI_API_KEY},
			json=payload,
			timeout=30,
		)
		if resp.status_code != 200:
			return "Gemini API error %d: %s" % (resp.status_code, resp.text[:500])

		data = resp.json()
		candidates = data.get("candidates", [])
		if not candidates:
			return "Gemini API returned no candidates."

		first = candidates[0].get("content", {})
		parts = first.get("parts", [])
		texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]

		if not texts:
			return "Gemini API returned an empty response."

		return "\n".join(texts)
	except Exception as e:
		return "There was an error while contacting Gemini: %s" % e

def render_sidebar_chat():
	"""Show a simple chat box in the sidebar that talks to Gemini."""

	#saves chat history while navigating between pages
	if "chat_history" not in st.session_state:
		st.session_state["chat_history"] = []

	st.markdown("### 💬 Chat Assistant")

	if not GEMINI_API_KEY:
		st.warning("Gemini API key is not configured. Set GOOGLE_API_KEY in your .env file.")

	if not st.session_state["chat_history"]:
		uploaded_file = st.file_uploader("Upload Chat History", type=["txt"], key="chat_history_uploader")
		
		if uploaded_file is not None:
			try:
				# Read and decode the text file
				stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
				chat_text = stringio.read()
				
				parsed_history = []
				# Split by the specific markers we use in the download function
				# Warning: This is a fragile parsing method if the user types "**You:** " in their message
				parts = chat_text.split("**You:** ")
				
				for part in parts:
					p = part.strip()
					if not p:
						continue
					
					if "**Assistant:** " in p:
						# Split into user and assistant part
						user_part, assistant_part = p.split("**Assistant:** ", 1)
						parsed_history.append(("user", user_part.strip()))
						parsed_history.append(("assistant", assistant_part.strip()))
					else:
						# Just user message? or malformed
						pass
				
				if parsed_history and parsed_history != st.session_state["chat_history"]:
					st.session_state["chat_history"] = parsed_history
					st.success("Chat history restored from text file!")
					st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
				elif not parsed_history:
					st.warning("Could not find recognized chat format in file.")

			except Exception as e:
				st.error(f"Error loading history: {e}")
		

	# Show previous messages in a collapsible, 300 size scrollable box
	if st.session_state["chat_history"]:
		with st.expander("View History", expanded=True):
			with st.container(height=300):
				for role, content in st.session_state["chat_history"]:
					if role == "user":
						st.markdown("**You:** %s" % content)
					else:
						st.markdown("**Assistant:** %s" % content)

	user_input = st.text_area(
		"Ask a question about your current analysis",
		key="gemini_chat_input",
		height=80,
	)

	# Function to handle sending message correctly and clearing the input
	def handle_send():
		text = st.session_state.get("gemini_chat_input", "").strip()
		if text:
			# Save user question
			st.session_state["chat_history"].append(("user", text))
			# Get answer from Gemini
			answer = call_gemini_with_context(text)
			st.session_state["chat_history"].append(("assistant", answer))
			# Clear the input box
			st.session_state["gemini_chat_input"] = ""

	if st.button("Send", key="gemini_chat_send", use_container_width=True, on_click=handle_send):
		# logic handled in callback to allow clearing st.session_state["gemini_chat_input"]
		pass

	st.markdown("---")

	if st.session_state["chat_history"]:
		col1, col2 = st.columns(2)
		if col1.button("Clear History", key="gemini_chat_clear", use_container_width=True):
			st.session_state["chat_history"] = []
			st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

		with col2: 
			# Export as Text to be compatible with the upload functionality
			history_text = ""
			for role, content in st.session_state["chat_history"]:
				if role == "user":
					history_text += "**You:** %s\n\n" % content
				else:
					history_text += "**Assistant:** %s\n\n" % content
			st.download_button("Download Chat History", history_text, "chat_history.txt")


