import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from scripts.utils import load_local_files, load_uploaded_files

# --- Page Configuration ---
st.set_page_config(page_title="Interactive Statistical Tests", layout="wide")
st.title("Interactive Statistical Tests")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Selection")
    camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]
    selected_camera = st.selectbox("Select Camera", camera_names)
    
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    
    base_dir = os.path.join("timeseries", selected_camera)
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    csv_files = [os.path.relpath(f, base_dir) for f in csv_files]
    selected_csvs = st.multiselect("Select CSV Files", csv_files)

# --- Load Data ---
dfs = []
if selected_csvs:
    dfs.extend(load_local_files(base_dir, selected_csvs))
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))
    
if not dfs:
    st.warning("Please select or upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

if data.empty:
    st.warning("No data found in the files.")
    st.stop()

# --- Preprocess Columns Based on CSV Header ---
# Your CSV header looks like:
# File,Timestamp,Anoplopoma,Asteroidea,Bubble,Chionoecetes,Eptatretus,
# Euphausia,Liponema,Microstomus,Sebastes,Zoarcidae,Cluster 0,Cluster 1,
# Cluster 2,Cluster 3,Temperature,Conductivity,Pressure,Salinity,
# "Oxygen Phase, usec",Oxygen Temperature Voltage,PressurePSI
#
# Exclude "File" and "Timestamp" from analysis.
exclude_cols = ["File", "Timestamp"]
# Identify numeric columns (e.g., species counts and environmental metrics)
numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
# Identify categorical columns from the remaining ones.
categorical_cols = [col for col in data.columns if col not in numeric_cols and col not in exclude_cols]

st.info("Note: If any numeric column (such as cluster labels) should be treated as categorical, please convert it to string before analysis.")

# --- Select Statistical Test ---
st.subheader("Choose a Statistical Test")
test_options = [
    "Independent T-Test", "Paired T-Test", "Chi-Square Test", "Mann-Whitney U Test",
    "One-Way ANOVA", "Kruskal-Wallis Test", "Pearson Correlation", "Spearman Correlation",
    "Fisher's Exact Test"
]
selected_test = st.selectbox("Select Test", test_options)

# Prepare a string to capture results for export.
result_str = ""

# ----------------- Independent T-Test ----------------- #
if selected_test == "Independent T-Test":
    st.write("### Independent T-Test")
    if not numeric_cols or not categorical_cols:
        st.error("Dataset must have both numeric and categorical variables for this test.")
    else:
        num_var = st.selectbox("Select Numeric Variable", numeric_cols)
        cat_var = st.selectbox("Select Grouping Variable (exactly 2 groups)", categorical_cols)
        groups = data[cat_var].dropna().unique()
        if len(groups) != 2:
            st.error(f"Grouping variable '{cat_var}' has {len(groups)} groups; exactly 2 groups are required.")
        else:
            group1 = data[data[cat_var] == groups[0]][num_var].dropna()
            group2 = data[data[cat_var] == groups[1]][num_var].dropna()
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            result_str = (
                f"Independent T-Test for **{num_var}** grouped by **{cat_var}**:\n"
                f"- Group 1 ({groups[0]}): mean = {group1.mean():.3f}\n"
                f"- Group 2 ({groups[1]}): mean = {group2.mean():.3f}\n"
                f"- T-statistic = {t_stat:.3f}, p-value = {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: Statistically significant difference between the two groups."
            else:
                result_str += "Result: No statistically significant difference between the groups."
            st.text(result_str)

# ----------------- Paired T-Test ----------------- #
elif selected_test == "Paired T-Test":
    st.write("### Paired T-Test")
    if len(numeric_cols) < 2:
        st.error("At least two numeric variables are required for a Paired T-Test.")
    else:
        pair1 = st.selectbox("Select First Variable", numeric_cols)
        pair2 = st.selectbox("Select Second Variable", [col for col in numeric_cols if col != pair1])
        paired_data = data[[pair1, pair2]].dropna()
        t_stat, p_val = stats.ttest_rel(paired_data[pair1], paired_data[pair2])
        mean_diff = (paired_data[pair1] - paired_data[pair2]).mean()
        result_str = (
            f"Paired T-Test for **{pair1}** and **{pair2}**:\n"
            f"- Mean difference = {mean_diff:.3f}\n"
            f"- T-statistic = {t_stat:.3f}, p-value = {p_val:.3f}\n\n"
        )
        if p_val < 0.05:
            result_str += "Result: Statistically significant difference between the paired variables."
        else:
            result_str += "Result: No statistically significant difference between the paired variables."
        st.text(result_str)

# ----------------- Chi-Square Test ----------------- #
elif selected_test == "Chi-Square Test":
    st.write("### Chi-Square Test")
    if len(categorical_cols) < 2:
        st.error("At least two categorical variables are required for a Chi-Square Test.")
    else:
        cat_var1 = st.selectbox("Select First Categorical Variable", categorical_cols)
        cat_var2 = st.selectbox("Select Second Categorical Variable", [col for col in categorical_cols if col != cat_var1])
        contingency = pd.crosstab(data[cat_var1], data[cat_var2])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        result_str = (
            f"Chi-Square Test for **{cat_var1}** vs **{cat_var2}**:\n"
            f"- Chi² = {chi2:.3f}, p-value = {p_val:.3f}, Degrees of Freedom = {dof}\n\n"
        )
        if p_val < 0.05:
            result_str += "Result: Statistically significant association between the variables."
        else:
            result_str += "Result: No statistically significant association between the variables."
        st.text(result_str)
        st.write("Contingency Table:")
        st.dataframe(contingency)

# ----------------- Mann-Whitney U Test ----------------- #
elif selected_test == "Mann-Whitney U Test":
    st.write("### Mann-Whitney U Test")
    if not numeric_cols or not categorical_cols:
        st.error("Dataset must have both numeric and categorical variables for this test.")
    else:
        num_var = st.selectbox("Select Numeric Variable", numeric_cols)
        cat_var = st.selectbox("Select Grouping Variable (exactly 2 groups)", categorical_cols)
        groups = data[cat_var].dropna().unique()
        if len(groups) != 2:
            st.error(f"Grouping variable '{cat_var}' has {len(groups)} groups; exactly 2 groups are required.")
        else:
            group1 = data[data[cat_var] == groups[0]][num_var].dropna()
            group2 = data[data[cat_var] == groups[1]][num_var].dropna()
            u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            result_str = (
                f"Mann-Whitney U Test for **{num_var}** grouped by **{cat_var}**:\n"
                f"- Group 1 ({groups[0]}): median = {group1.median():.3f}\n"
                f"- Group 2 ({groups[1]}): median = {group2.median():.3f}\n"
                f"- U-statistic = {u_stat:.3f}, p-value = {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: Statistically significant difference exists between the groups."
            else:
                result_str += "Result: No statistically significant difference exists between the groups."
            st.text(result_str)

# ----------------- One-Way ANOVA ----------------- #
elif selected_test == "One-Way ANOVA":
    st.write("### One-Way ANOVA")
    if not numeric_cols or not categorical_cols:
        st.error("Dataset must have both numeric and categorical variables for this test.")
    else:
        num_var = st.selectbox("Select Numeric Variable", numeric_cols)
        cat_var = st.selectbox("Select Grouping Variable", categorical_cols)
        groups = data[cat_var].dropna().unique()
        if len(groups) < 2:
            st.error(f"Grouping variable '{cat_var}' must have at least 2 groups (found {len(groups)}).")
        else:
            group_data = [data[data[cat_var] == grp][num_var].dropna() for grp in groups]
            f_stat, p_val = stats.f_oneway(*group_data)
            result_str = (
                f"One-Way ANOVA for **{num_var}** by **{cat_var}**:\n"
                f"- F-statistic = {f_stat:.3f}, p-value = {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: Statistically significant differences exist between the group means."
            else:
                result_str += "Result: No statistically significant differences between the group means."
            st.text(result_str)

# ----------------- Kruskal-Wallis Test ----------------- #
elif selected_test == "Kruskal-Wallis Test":
    st.write("### Kruskal-Wallis Test")
    if not numeric_cols or not categorical_cols:
        st.error("Dataset must have both numeric and categorical variables for this test.")
    else:
        num_var = st.selectbox("Select Numeric Variable", numeric_cols)
        cat_var = st.selectbox("Select Grouping Variable", categorical_cols)
        groups = data[cat_var].dropna().unique()
        if len(groups) < 2:
            st.error(f"Grouping variable '{cat_var}' must have at least 2 groups (found {len(groups)}).")
        else:
            group_data = [data[data[cat_var] == grp][num_var].dropna() for grp in groups]
            h_stat, p_val = stats.kruskal(*group_data)
            result_str = (
                f"Kruskal-Wallis Test for **{num_var}** by **{cat_var}**:\n"
                f"- H-statistic = {h_stat:.3f}, p-value = {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: Statistically significant differences exist among the groups."
            else:
                result_str += "Result: No statistically significant differences exist among the groups."
            st.text(result_str)

# ----------------- Pearson Correlation ----------------- #
elif selected_test == "Pearson Correlation":
    st.write("### Pearson Correlation")
    if len(numeric_cols) < 2:
        st.error("At least two numeric variables are required for correlation analysis.")
    else:
        var1 = st.selectbox("Select First Numeric Variable", numeric_cols)
        var2 = st.selectbox("Select Second Numeric Variable", [col for col in numeric_cols if col != var1])
        paired_data = data[[var1, var2]].dropna()
        corr_coef, p_val = stats.pearsonr(paired_data[var1], paired_data[var2])
        result_str = (
            f"Pearson Correlation between **{var1}** and **{var2}**:\n"
            f"- Correlation coefficient = {corr_coef:.3f}, p-value = {p_val:.3f}\n\n"
        )
        if p_val < 0.05:
            result_str += "Result: A statistically significant linear relationship exists between the variables."
        else:
            result_str += "Result: No statistically significant linear relationship exists between the variables."
        st.text(result_str)

# ----------------- Spearman Correlation ----------------- #
elif selected_test == "Spearman Correlation":
    st.write("### Spearman Correlation")
    if len(numeric_cols) < 2:
        st.error("At least two numeric variables are required for correlation analysis.")
    else:
        var1 = st.selectbox("Select First Numeric Variable", numeric_cols)
        var2 = st.selectbox("Select Second Numeric Variable", [col for col in numeric_cols if col != var1])
        paired_data = data[[var1, var2]].dropna()
        corr_coef, p_val = stats.spearmanr(paired_data[var1], paired_data[var2])
        result_str = (
            f"Spearman Correlation between **{var1}** and **{var2}**:\n"
            f"- Correlation coefficient = {corr_coef:.3f}, p-value = {p_val:.3f}\n\n"
        )
        if p_val < 0.05:
            result_str += "Result: A statistically significant monotonic relationship exists between the variables."
        else:
            result_str += "Result: No statistically significant monotonic relationship exists between the variables."
        st.text(result_str)

# ----------------- Fisher's Exact Test ----------------- #
elif selected_test == "Fisher's Exact Test":
    st.write("### Fisher's Exact Test")
    if len(categorical_cols) < 2:
        st.error("At least two categorical variables are required for Fisher's Exact Test.")
    else:
        cat_var1 = st.selectbox("Select First Categorical Variable", categorical_cols)
        cat_var2 = st.selectbox("Select Second Categorical Variable", [col for col in categorical_cols if col != cat_var1])
        contingency = pd.crosstab(data[cat_var1], data[cat_var2])
        if contingency.shape != (2, 2):
            st.error("Fisher's Exact Test is only applicable for 2x2 contingency tables. Please select variables that yield a 2x2 table.")
        else:
            oddsratio, p_val = stats.fisher_exact(contingency)
            result_str = (
                f"Fisher's Exact Test for **{cat_var1}** vs **{cat_var2}**:\n"
                f"- Odds Ratio = {oddsratio:.3f}, p-value = {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: A statistically significant association exists between the variables."
            else:
                result_str += "Result: No statistically significant association exists between the variables."
            st.text(result_str)
            st.write("Contingency Table:")
            st.dataframe(contingency)

# ----------------- Export Results ----------------- #
if result_str:
    st.subheader("Export Results")
    export_format = st.selectbox("Select Export Format", ["Text (.txt)", "CSV (.csv)"])
    if export_format == "Text (.txt)":
        export_data = result_str
        export_filename = "statistical_test_results.txt"
        mime_type = "text/plain"
    else:
        export_data = pd.DataFrame({"Result": [result_str]})
        export_filename = "statistical_test_results.csv"
        mime_type = "text/csv"
    
    st.download_button(
        label="Download Results",
        data=export_data if isinstance(export_data, str) else export_data.to_csv(index=False),
        file_name=export_filename,
        mime=mime_type
    )
