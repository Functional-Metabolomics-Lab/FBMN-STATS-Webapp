import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

def run_random_forest(data, md, attribute, n_trees, random_seed=None, _progress_callback=None):
    # initialize a log to print out in the app later
    log = ""

    labels = md[[attribute]]

    # Check for NaN in labels (y)
    if labels.isnull().values.any():
        raise ValueError("Input y contains NaN. Please remove or impute missing values in your class/attribute column before running the model.")

    # Change the values of the attribute of interest from strings to a numerical
    enc = OrdinalEncoder()
    # st.write(labels.value_counts()) # Displays the sample size for each group
    labels = enc.fit_transform(labels)
    labels = np.array([x[0] + 1 for x in labels])

    class_names = [str(c).strip() for c in enc.categories_[0]]

    # Extract the feature intensities as np 2D array
    features = np.array(data)

    # Determine the smallest class size and adjust test_size accordingly
    unique, counts = np.unique(labels, return_counts=True)
    min_test_size = float(len(unique)) / len(labels)

    # Adjust test size to be larger of the calculated min_test_size or the initial_test_size
    adjusted_test_size = max(min_test_size, 0.20)
    
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=adjusted_test_size, random_state=random_seed, stratify=labels)

    # Collecting info about feature and label shapes for logging
    log += f"Class names: {', '.join(class_names)}\n"
    log += f"Train/Test split: {1 - adjusted_test_size:.0%} / {adjusted_test_size:.0%}\n"
    log += f"Training Features Shape: {train_features.shape}\n"
    log += f"Testing Features Shape: {test_features.shape}\n"
    for cls_val, cls_name in zip(sorted(np.unique(labels)), class_names):
        n_train = np.sum(train_labels == cls_val)
        n_test = np.sum(test_labels == cls_val)
        log += f"{cls_name}: {n_train} train, {n_test} test\n"

    # Balance the weights of the attribute of interest to account for unbalanced sample sizes per group
    sklearn_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels)
    
    weights = {}
    for i,w in enumerate(np.unique(train_labels)):
        weights[w] = sklearn_weights[i]

    # Set up the random forest classifier with 100 tress, balanded weights, and a random state to make it reproducible
    rf = RandomForestClassifier(n_estimators=n_trees, class_weight= weights, random_state=random_seed)
   
    # Fit the classifier to the training set
    rf.fit(train_features, train_labels)

    # Use the random forest classifier to predict the sample areas in the test set
    predictions_test = rf.predict(test_features)
    predictions_train = rf.predict(train_features)

    classifier_accuracy = round(rf.score(test_features, test_labels)*100, 2)
    log += f"Classifier mean accuracy score: {classifier_accuracy}%.\n"

    # Calculate confusion matrices using actual encoded label values (1-based)
    label_values = sorted(np.unique(labels))
    test_confusion_matrix = confusion_matrix(test_labels, predictions_test, labels=label_values)
    train_confusion_matrix = confusion_matrix(train_labels, predictions_train, labels=label_values)

    test_confusion_df = pd.DataFrame(test_confusion_matrix, index=class_names, columns=class_names)
    train_confusion_df = pd.DataFrame(train_confusion_matrix, index=class_names, columns=class_names)

    test_accuracy = accuracy_score(test_labels, predictions_test)
    train_accuracy = accuracy_score(train_labels, predictions_train)
    
    # Report of the accuracy of predictions on the test set
    class_report = classification_report(test_labels, predictions_test)

    # Print the sample areas corresponding to the numbers in the report
    label_mapping = "\n".join([f"{i+1.0} ,{cat}" for i, cat in enumerate(enc.categories_[0])])

    # Most important model quality plot
    # OOB error lines should flatline. If it doesn't flatline add more trees
    rf_oob = RandomForestClassifier(class_weight=weights, warm_start=True, oob_score=True, random_state=random_seed)
    errors = []
    tree_range = np.arange(1,500, 10)
    import time as _time
    _start = _time.time()
    for idx, i in enumerate(tree_range):
        rf_oob.set_params(n_estimators=i)
        rf_oob.fit(train_features, train_labels)
        errors.append(1-rf_oob.oob_score_)
        if _progress_callback is not None:
            elapsed = _time.time() - _start
            est_left = (elapsed / (idx + 1)) * (len(tree_range) - idx - 1) if idx > 0 else 0
            _progress_callback(idx + 1, len(tree_range), est_left)


    df_oob = pd.DataFrame({"n trees": tree_range, "error rate": errors})

    # OOB evaluation on all samples, as in the protocol (rfPermute, Steps 53-55): every tree is
    # validated on the samples it did not see, with class-balanced sampling, and no separate test split.
    rf_all = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced_subsample",
                                    oob_score=True, random_state=random_seed)
    rf_all.fit(features, labels)
    oob_proba = rf_all.oob_decision_function_
    has_oob = ~np.isnan(oob_proba).any(axis=1)
    oob_pred = rf_all.classes_[np.argmax(np.nan_to_num(oob_proba, nan=-1.0), axis=1)]
    oob_confusion = confusion_matrix(labels[has_oob], oob_pred[has_oob], labels=label_values)
    oob_confusion_df = pd.DataFrame(oob_confusion, index=class_names, columns=class_names)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_correct = np.diag(oob_confusion) / oob_confusion.sum(axis=1) * 100
    oob_confusion_df["pct.correct"] = np.round(pct_correct, 1)
    oob_accuracy = accuracy_score(labels[has_oob], oob_pred[has_oob])
    log += f"OOB accuracy (all samples, class-balanced, {n_trees} trees): {oob_accuracy:.1%}\n"

    # Extract the important features in the model
    df_important_features = pd.DataFrame(rf.feature_importances_, 
                                         index=data.columns).sort_values(by=0, ascending=False)
    df_important_features.columns = ["importance"]
    
    return df_oob, df_important_features, log, class_report, label_mapping, test_confusion_df, train_confusion_df, test_accuracy, train_accuracy, oob_confusion_df, oob_accuracy


def get_oob_fig(df):
    return px.line(df, x="n trees", y="error rate", title="out-of-bag (OOB) error")


def get_feature_importance_fig(df_important_features, n_features):
    """Return a horizontal bar chart of the top *n_features* by Gini importance."""
    top = df_important_features.head(n_features).iloc[::-1]  # reverse for horizontal bar
    full_labels = list(top.index.astype(str))
    truncated_labels = [s[:30] + "…" if len(s) > 30 else s for s in full_labels]
    # Plot against the full (unique) labels so features sharing a 30-character
    # prefix don't collapse into one bar; show the truncated text as tick labels.
    fig = px.bar(
        top,
        x="importance",
        y=full_labels,
        orientation="h",
        template="plotly_white",
        title=f"Top {n_features} Features by Importance",
        labels={"importance": "Gini Importance", "y": "Feature"},
    )
    fig.update_yaxes(tickmode="array", tickvals=full_labels, ticktext=truncated_labels)
    fig.update_layout(
        font={"color": "grey", "size": 12, "family": "Sans"},
        title={"font_color": "#3E3D53"},
        yaxis_title="",
        height=max(400, n_features * 22),
    )
    return fig

def classification_report_to_df(report):
    
    # Split the report into lines
    lines = report.split("\n")
    
    # Prepare a dictionary to hold the data
    report_data = {"class": [], "precision": [], "recall": [], "f1-score": [], "support": []}
    
    for line in lines[2:-3]:  # Skip the header and summary lines
        parts = line.split()
        # Ensure that the line contains the expected number of parts
        if len(parts) == 5:
            report_data["class"].append(parts[0])
            report_data["precision"].append(parts[1])
            report_data["recall"].append(parts[2])
            report_data["f1-score"].append(parts[3])
            report_data["support"].append(parts[4])
    
    # Convert the dictionary to a DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Convert numeric columns from strings to floats
    report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].astype(float)
    report_df["support"] = report_df["support"].astype(int)
    
    return report_df

def label_mapping_to_df(label_mapping_str):
    
    # Split the string into lines
    lines = label_mapping_str.split("\n")
    
    # Split each line into index and label, then collect into a list of tuples
    mapping = [line.split(" ,") for line in lines if line]  # Ensure the line is not empty
    
    # Convert the list of tuples into a DataFrame
    mapping_df = pd.DataFrame(mapping, columns=['class', 'Label'])
    mapping_df['class'] = mapping_df['class'].astype(str)
    return mapping_df


