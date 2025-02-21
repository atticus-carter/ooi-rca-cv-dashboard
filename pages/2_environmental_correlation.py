import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns
import os
import glob
import matplotlib.pyplot as plt  # For some plots

# --- Page Configuration ---
st.set_page_config(page_title="Environmental Correlation Analysis", layout="wide")
st.title("Environmental Correlation Analysis")

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
    
# --- Load and Process Data ---
dfs = []
if selected_csvs:
    dfs.extend(load_local_files(base_dir, selected_csvs))
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))

if not dfs:
    st.warning("Please select or upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

if 'Timestamp' in data.columns:
    # Convert Timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Extract columns (species, clusters, environmental variables)
    class_names, cluster_cols, env_vars = extract_data_columns(data)
    
    # --- Option to include/deselect Bubble as a species ---
    if "Bubble" in class_names:
        include_bubble = st.sidebar.checkbox("Include 'Bubble' as a species", value=False)
        if not include_bubble:
            class_names = [sp for sp in class_names if sp != "Bubble"]
    
    # --- Analysis Type Selection ---
    analysis_options = [
        "Correlation Matrix",
        "Environmental Response Curves",
        "PCA Analysis",
        "Time-lagged Correlations",
        "Threshold Analysis",
        "Random Forest Regression Analysis",
        "Linear Regression Analysis",
        "ANOVA Analysis",
        "Mann-Whitney U Test Analysis",
        "Time Series Forecasting (ARIMA)",
        "K-Means Clustering Analysis",
        "Hierarchical Clustering Dendrogram",
        "Rolling Window Correlation Analysis",
        "Fourier Transform Analysis",
        "Granger Causality Test",
        "Decision Tree Analysis",
        "Scatter Matrix Analysis",
        "Canonical Correlation Analysis",
        "LOESS Smoothing Analysis",
        "Outlier Detection Analysis",
        "Change Point Detection Analysis",
        "Neural Network Regression Analysis",
        "Cross-Validation Model Analysis",
        "Bayesian Regression Analysis",
        "Bubble Outgassing Analysis"
    ]
    
    analysis_type = st.selectbox("Select Analysis Type", analysis_options)
    
    # ------------------------ Existing Analyses ------------------------ #
    if analysis_type == "Correlation Matrix":
        # Create correlation matrix between environmental variables and species
        corr_df = data[env_vars + class_names].corr()
        
        # Filter to show only env vars vs species correlations
        env_species_corr = corr_df.loc[env_vars, class_names]
        
        fig = px.imshow(
            env_species_corr,
            title="Environment-Species Correlation Matrix",
            labels=dict(x="Species", y="Environmental Variables", color="Correlation"),
            aspect="auto",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistical significance
        st.subheader("Statistical Significance")
        p_values = pd.DataFrame(index=env_vars, columns=class_names)
        for env in env_vars:
            for species in class_names:
                _, p_val = stats.pearsonr(data[env], data[species])
                p_values.loc[env, species] = p_val
        
        # Display significant correlations
        sig_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
        significant_corr = (p_values < sig_level) & (abs(env_species_corr) > 0.3)
        if significant_corr.any().any():
            st.write("Significant correlations (p < {:.2f}):".format(sig_level))
            for env in env_vars:
                for species in class_names:
                    if significant_corr.loc[env, species]:
                        st.write(f"{env} vs {species}: r = {env_species_corr.loc[env, species]:.3f}")
        st.write("Plain Language Summary: The correlation matrix displays the linear relationships between environmental variables and species abundances. Statistically significant correlations (with p-values below the chosen significance level) are highlighted, suggesting variables that may be interrelated.")

    elif analysis_type == "Environmental Response Curves":
        col1, col2 = st.columns(2)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            species = st.selectbox("Select Species", class_names)
        
        # Create scatter plot with trend line
        fig = px.scatter(
            data, 
            x=env_var, 
            y=species,
            trendline="lowess",
            title=f"{species} Response to {env_var}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add density distribution
        fig2 = go.Figure()
        fig2.add_histogram2d(
            x=data[env_var],
            y=data[species],
            colorscale="Viridis",
            nbinsx=30,
            nbinsy=30
        )
        fig2.update_layout(title=f"Density Distribution: {species} vs {env_var}")
        st.plotly_chart(fig2, use_container_width=True)
        st.write("Plain Language Summary: The response curves reveal how the abundance of a selected species changes with variations in an environmental parameter. The trendline and density plot together highlight underlying patterns.")

    elif analysis_type == "PCA Analysis":
        # Standardize environmental variables
        scaler = StandardScaler()
        env_scaled = scaler.fit_transform(data[env_vars])
        
        # Perform PCA
        pca = PCA()
        env_pca = pca.fit_transform(env_scaled)
        
        # Create scree plot
        explained_variance = pca.explained_variance_ratio_ * 100
        fig = px.line(
            x=range(1, len(explained_variance) + 1),
            y=explained_variance,
            markers=True,
            title="PCA Scree Plot",
            labels={"x": "Principal Component", "y": "Explained Variance (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show PCA loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(env_vars))],
            index=env_vars
        )
        st.write("PCA Loadings:")
        st.dataframe(loadings)
        st.write("Plain Language Summary: PCA reduces the dimensionality of environmental data. The scree plot shows how much variance each principal component explains, and the loadings indicate which variables contribute most to each component.")

    elif analysis_type == "Time-lagged Correlations":
        col1, col2, col3 = st.columns(3)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            species = st.selectbox("Select Species", class_names)
        with col3:
            max_lag = st.slider("Maximum Lag (hours)", 1, 48, 24)
        
        # Calculate lagged correlations
        correlations = []
        for lag in range(max_lag + 1):
            corr = data[env_var].shift(lag).corr(data[species])
            correlations.append({"lag": lag, "correlation": corr})
        
        lag_df = pd.DataFrame(correlations)
        fig = px.line(
            lag_df,
            x="lag",
            y="correlation",
            title=f"Time-lagged Correlation: {species} vs {env_var}",
            labels={"lag": "Lag (hours)", "correlation": "Correlation Coefficient"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Plain Language Summary: This plot shows how the correlation between the chosen environmental variable and species abundance changes with time lag. It can reveal delays in response between environmental changes and biological reactions.")

    elif analysis_type == "Threshold Analysis":
        col1, col2 = st.columns(2)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            species = st.selectbox("Select Species", class_names)
        
        # Calculate percentiles for threshold analysis
        percentiles = np.percentile(data[env_var], [25, 50, 75])
        thresholds = {
            "Low": (data[env_var] <= percentiles[0]),
            "Medium": (data[env_var] > percentiles[0]) & (data[env_var] <= percentiles[2]),
            "High": (data[env_var] > percentiles[2])
        }
        
        # Calculate species statistics for each threshold
        threshold_stats = []
        for threshold_name, mask in thresholds.items():
            stats_dict = {
                "threshold": threshold_name,
                "mean": data.loc[mask, species].mean(),
                "std": data.loc[mask, species].std(),
                "count": mask.sum()
            }
            threshold_stats.append(stats_dict)
        
        stats_df = pd.DataFrame(threshold_stats)
        fig = px.bar(
            stats_df,
            x="threshold",
            y="mean",
            error_y="std",
            title=f"{species} Abundance by {env_var} Thresholds",
            labels={"threshold": f"{env_var} Level", "mean": f"Mean {species} Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Plain Language Summary: Threshold analysis splits the environmental variable into ranges (low, medium, high) and compares the species’ abundance across these groups. Differences in means and variability can indicate threshold effects.")

    # ---------------------- New Analysis Options ---------------------- #
    elif analysis_type == "Random Forest Regression Analysis":
        # Use Random Forest to predict a species from environmental variables
        target = st.selectbox("Select Target Species", class_names)
        predictors = env_vars
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        X = data[predictors].dropna()
        y = data[target].loc[X.index]
        rf.fit(X, y)
        importance = rf.feature_importances_
        imp_df = pd.DataFrame({"Feature": predictors, "Importance": importance})
        fig = px.bar(imp_df.sort_values("Importance", ascending=False), x="Feature", y="Importance",
                     title=f"Random Forest Feature Importances for predicting {target}")
        st.plotly_chart(fig, use_container_width=True)
        r2 = rf.score(X, y)
        st.write(f"Plain Language Summary: The Random Forest model explains approximately {r2*100:.1f}% of the variance in {target}. Higher feature importance values indicate more influential predictors among the environmental variables.")

    elif analysis_type == "Linear Regression Analysis":
        col1, col2 = st.columns(2)
        with col1:
            predictor = st.selectbox("Select Predictor Variable", env_vars)
        with col2:
            target = st.selectbox("Select Target Species", class_names)
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[predictor], data[target])
        fig = px.scatter(data, x=predictor, y=target, title=f"Linear Regression: {target} vs {predictor}")
        fig.add_traces(go.Scatter(x=data[predictor], y=intercept + slope*data[predictor],
                                  mode='lines', name='Fit'))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: The regression indicates that for each unit increase in {predictor}, {target} changes by {slope:.3f} units on average. Approximately {r_value**2*100:.1f}% of the variance is explained (p-value = {p_value:.3f}).")

    elif analysis_type == "ANOVA Analysis":
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Species", class_names)
        with col2:
            group_var = st.selectbox("Select Grouping Variable (Environmental)", env_vars)
        data['group'] = pd.qcut(data[group_var], q=4, labels=["Q1","Q2","Q3","Q4"])
        groups = [group[target].dropna() for name, group in data.groupby('group')]
        f_val, p_val = stats.f_oneway(*groups)
        fig = px.box(data, x='group', y=target, title=f"ANOVA: {target} by quartiles of {group_var}")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: ANOVA produced an F-value of {f_val:.2f} and a p-value of {p_val:.3f}. This suggests that differences in {target} among groups defined by {group_var} are {'statistically significant' if p_val < 0.05 else 'not statistically significant'}.")

    elif analysis_type == "Mann-Whitney U Test Analysis":
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Species", class_names)
        with col2:
            group_var = st.selectbox("Select Binary Grouping Variable", env_vars)
        median_val = data[group_var].median()
        group1 = data[data[group_var] <= median_val][target].dropna()
        group2 = data[data[group_var] > median_val][target].dropna()
        u_stat, p_val = stats.mannwhitneyu(group1, group2)
        fig = px.box(data, x=(data[group_var] > median_val).astype(str), y=target, 
                     title=f"Mann-Whitney U: {target} split by median of {group_var}")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: The Mann-Whitney U test statistic is {u_stat:.2f} with a p-value of {p_val:.3f}. This indicates that the difference in {target} between the two groups split by {group_var} is {'statistically significant' if p_val < 0.05 else 'not statistically significant'}.")

    elif analysis_type == "Time Series Forecasting (ARIMA)":
        import statsmodels.api as sm
        target = st.selectbox("Select Time Series Variable", class_names + env_vars)
        order = st.text_input("Enter ARIMA order (p,d,q)", value="(1,1,1)")
        order = eval(order)
        ts_data = data.set_index('timestamp')[target].dropna()
        model = sm.tsa.ARIMA(ts_data, order=order)
        results = model.fit()
        forecast = results.get_forecast(steps=10)
        forecast_index = pd.date_range(ts_data.index[-1], periods=11, closed='right')
        forecast_df = forecast.summary_frame()
        fig = px.line(x=ts_data.index, y=ts_data, title=f"ARIMA Forecast for {target}")
        fig.add_scatter(x=forecast_index, y=forecast_df['mean'], mode='lines', name='Forecast')
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: The ARIMA model with order {order} (AIC = {results.aic:.2f}) forecasts future values of {target}. The forecast curve and confidence intervals help visualize expected trends.")

    elif analysis_type == "K-Means Clustering Analysis":
        from sklearn.cluster import KMeans
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        X = data[env_vars].dropna()
        kmeans.fit(X)
        labels = kmeans.labels_
        X_cluster = X.copy()
        X_cluster['Cluster'] = labels
        fig = px.scatter_matrix(X_cluster, dimensions=env_vars, color="Cluster", title="K-Means Clustering Analysis")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: The environmental data was grouped into {n_clusters} clusters. These clusters may indicate natural groupings or regimes within the dataset.")

    elif analysis_type == "Hierarchical Clustering Dendrogram":
        from scipy.cluster.hierarchy import dendrogram, linkage
        X = data[env_vars].dropna()
        linked = linkage(X, 'ward')
        fig, ax = plt.subplots(figsize=(10, 7))
        dendrogram(linked, labels=X.index, ax=ax)
        st.pyplot(fig)
        st.write("Plain Language Summary: The dendrogram visualizes the hierarchical clustering of environmental data, where the branch lengths reflect the similarity between clusters.")

    elif analysis_type == "Rolling Window Correlation Analysis":
        col1, col2, col3 = st.columns(3)
        with col1:
            var1 = st.selectbox("Select First Variable", env_vars + class_names)
        with col2:
            var2 = st.selectbox("Select Second Variable", env_vars + class_names)
        with col3:
            window_size = st.slider("Select window size (data points)", 10, 100, 30)
        roll_corr = data[var1].rolling(window=window_size).corr(data[var2])
        fig = px.line(x=data.index, y=roll_corr, title=f"Rolling Window Correlation: {var1} & {var2}")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Plain Language Summary: This rolling window analysis shows how the correlation between the two selected variables evolves over time, which can reveal periods of stronger or weaker association.")

    elif analysis_type == "Fourier Transform Analysis":
        var = st.selectbox("Select Variable for FFT Analysis", env_vars + class_names)
        ts = data[var].dropna()
        fft_vals = np.fft.fft(ts)
        fft_freq = np.fft.fftfreq(len(ts))
        fig = px.line(x=fft_freq, y=np.abs(fft_vals), title=f"FFT Analysis for {var}", labels={"x": "Frequency", "y": "Amplitude"})
        st.plotly_chart(fig, use_container_width=True)
        st.write("Plain Language Summary: The Fourier Transform analysis identifies dominant frequency components in the selected variable, suggesting underlying periodic or cyclical behavior.")

    elif analysis_type == "Granger Causality Test":
        import statsmodels.tsa.stattools as tsastat
        col1, col2 = st.columns(2)
        with col1:
            cause = st.selectbox("Select 'Cause' Variable", env_vars + class_names)
        with col2:
            effect = st.selectbox("Select 'Effect' Variable", env_vars + class_names)
        maxlag = st.slider("Select Maximum Lag", 1, 10, 3)
        test_result = tsastat.grangercausalitytests(data[[effect, cause]].dropna(), maxlag=maxlag, verbose=False)
        p_vals = [round(test_result[i+1][0]['ssr_ftest'][1], 3) for i in range(maxlag)]
        st.write(f"Plain Language Summary: Granger causality test p-values for lags 1 to {maxlag} are: {p_vals}. A p-value below 0.05 at any lag suggests that {cause} may help predict {effect}.")
        
    elif analysis_type == "Decision Tree Analysis":
        from sklearn.tree import DecisionTreeRegressor, plot_tree
        target = st.selectbox("Select Target Species", class_names)
        predictors = env_vars
        X = data[predictors].dropna()
        y = data[target].loc[X.index]
        dt = DecisionTreeRegressor(random_state=42, max_depth=5)
        dt.fit(X, y)
        # Adjust figure size and spacing for better readability
        fig, ax = plt.subplots(figsize=(25, 10), dpi=300)
        plot_tree(dt, feature_names=predictors, filled=True, ax=ax, 
                 fontsize=12, precision=2, 
                 node_ids=True, max_depth=4)
        plt.margins(x=0.01)  # Reduce horizontal margins
        plt.tight_layout(pad=1.0)  # Add padding around the plot
        st.pyplot(fig)
        st.write("Plain Language Summary: The decision tree segments the data based on the environmental predictors to explain variations in the target species. The tree structure reflects the hierarchy of variable splits and their relative importance.")
        
    elif analysis_type == "Scatter Matrix Analysis":
        vars_selected = st.multiselect("Select variables for scatter matrix", env_vars + class_names, default=env_vars[:3] + class_names[:1])
        if vars_selected:
            fig = px.scatter_matrix(data, dimensions=vars_selected, title="Scatter Matrix Analysis")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Plain Language Summary: The scatter matrix displays pairwise relationships between the selected variables, offering a broad overview of potential correlations and patterns.")
    
    elif analysis_type == "Canonical Correlation Analysis":
        from sklearn.cross_decomposition import CCA
        st.write("Select two sets of variables for Canonical Correlation Analysis.")
        set1 = st.multiselect("Select Set 1 (Environmental Variables)", env_vars, default=env_vars)
        set2 = st.multiselect("Select Set 2 (Species)", class_names, default=class_names)
        if set1 and set2:
            cca = CCA(n_components=min(len(set1), len(set2)))
            X = data[set1].dropna()
            Y = data[set2].loc[X.index]
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            fig = px.scatter(x=X_c[:,0], y=Y_c[:,0], title="Canonical Correlation Analysis (First Canonical Variables)",
                             labels={"x": "Canonical Variable 1 (Set 1)", "y": "Canonical Variable 1 (Set 2)"})
            st.plotly_chart(fig, use_container_width=True)
            st.write("Plain Language Summary: Canonical Correlation Analysis finds the linear combinations of two variable sets that are most correlated. This helps reveal the hidden relationships between environmental factors and species dynamics.")
    
    elif analysis_type == "LOESS Smoothing Analysis":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        col1, col2 = st.columns(2)
        with col1:
            predictor = st.selectbox("Select Predictor Variable", env_vars)
        with col2:
            target = st.selectbox("Select Target Species", class_names)
        frac = st.slider("LOESS Smoothing Fraction", 0.1, 1.0, 0.3)
        loess_smoothed = lowess(data[target], data[predictor], frac=frac)
        fig = px.scatter(x=data[predictor], y=data[target], title=f"LOESS Smoothing: {target} vs {predictor}")
        fig.add_traces(go.Scatter(x=loess_smoothed[:,0], y=loess_smoothed[:,1], mode='lines', name='LOESS Fit'))
        st.plotly_chart(fig, use_container_width=True)
        st.write("Plain Language Summary: The LOESS smoothing curve captures the local trends in the data without assuming a global model, thereby highlighting nuanced relationships between the predictor and target variables.")
    
    elif analysis_type == "Outlier Detection Analysis":
        var = st.selectbox("Select Variable for Outlier Detection", env_vars + class_names)
        z_scores = np.abs(stats.zscore(data[var].dropna()))
        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
        outliers = data[var].dropna()[z_scores > threshold]
        fig = px.histogram(data, x=var, title=f"Outlier Detection for {var}")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: Using a Z-score threshold of {threshold}, {len(outliers)} outliers were detected in {var}. These outliers may represent unusual events or measurement errors.")
    
    elif analysis_type == "Change Point Detection Analysis":
        var = st.selectbox("Select Time Series Variable", env_vars + class_names)
        window = st.slider("Rolling window size", 10, 100, 30)
        ts = data.set_index('timestamp')[var].dropna()
        rolling_mean = ts.rolling(window=window).mean()
        diff = rolling_mean.diff().abs()
        threshold = st.slider("Change detection threshold", float(ts.min()), float(ts.max()), float(ts.mean()))
        change_points = diff[diff > threshold].index
        fig = px.line(x=ts.index, y=ts, title=f"Change Point Detection for {var}")
        fig.add_scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean')
        for cp in change_points:
            fig.add_vline(x=cp, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Plain Language Summary: {len(change_points)} potential change points were detected in {var}. These points may indicate shifts in the underlying process or regime changes.")
    
    elif analysis_type == "Neural Network Regression Analysis":
        from sklearn.neural_network import MLPRegressor
        target = st.selectbox("Select Target Species", class_names)
        predictors = env_vars
        X = data[predictors].dropna()
        y = data[target].loc[X.index]
        mlp = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, random_state=42)
        mlp.fit(X, y)
        r2 = mlp.score(X, y)
        st.write(f"Neural Network model R²: {r2:.2f}")
        st.write("Plain Language Summary: The Neural Network model predicts the target variable based on the environmental predictors. An R² of {:.2f} suggests that the model accounts for about {:.1f}% of the variance in {target}.".format(r2, r2*100))
    
    elif analysis_type == "Cross-Validation Model Analysis":
        from sklearn.model_selection import cross_val_score
        target = st.selectbox("Select Target Species", class_names)
        predictors = env_vars
        X = data[predictors].dropna()
        y = data[target].loc[X.index]
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y, cv=5)
        st.write(f"Cross-validation scores: {scores}")
        st.write("Plain Language Summary: The cross-validation analysis provides an estimate of model performance on unseen data. Scores range from {:.2f} to {:.2f}, with an average of {:.2f}.".format(scores.min(), scores.max(), scores.mean()))
    
    elif analysis_type == "Bayesian Regression Analysis":
        try:
            import pymc3 as pm
            target = st.selectbox("Select Target Species", class_names)
            predictor = st.selectbox("Select Predictor Variable", env_vars)
            y_data = data[target].dropna()
            x_data = data[predictor].loc[y_data.index]
            with pm.Model() as model:
                intercept = pm.Normal('intercept', mu=0, sigma=10)
                slope = pm.Normal('slope', mu=0, sigma=10)
                sigma = pm.HalfNormal('sigma', sigma=1)
                mu = intercept + slope * x_data
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)
                trace = pm.sample(1000, tune=1000, target_accept=0.95, progressbar=False)
            st.write(pm.summary(trace))
            st.write("Plain Language Summary: The Bayesian regression model estimates the relationship between {0} and {1}. The posterior distributions of the parameters reflect the uncertainty in these estimates.".format(predictor, target))
        except ImportError:
            st.error("PyMC3 is not installed. Bayesian Regression Analysis requires PyMC3.")
    
    elif analysis_type == "Bubble Outgassing Analysis":
        if "Bubble" not in data.columns:
            st.error("Bubble data not found in dataset.")
        else:
            bubble_var = "Bubble"
            # Look for environmental variables related to methane or gas
            candidate_env = [var for var in env_vars if "methane" in var.lower() or "gas" in var.lower()]
            if candidate_env:
                env_var = st.selectbox("Select Environmental Variable for Outgassing Analysis", candidate_env)
            else:
                env_var = st.selectbox("Select Environmental Variable for Outgassing Analysis", env_vars)
            corr, p_val = stats.pearsonr(data[bubble_var].dropna(), data[env_var].dropna())
            fig = px.scatter(data, x=bubble_var, y=env_var, title=f"Bubble vs {env_var} Analysis")
            # Fit and add a trendline
            coeffs = np.polyfit(data[bubble_var].dropna(), data[env_var].dropna(), 1)
            poly1d_fn = np.poly1d(coeffs)
            fig.add_traces(go.Scatter(x=data[bubble_var], y=poly1d_fn(data[bubble_var]),
                                      mode='lines', name='Fit'))
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Plain Language Summary: The correlation between bubble counts and {env_var} is {corr:.3f} (p = {p_val:.3f}). This suggests that bubble activity may {'be predictive of' if p_val < 0.05 else 'not be predictive of'} outgassing events.")

else:
    st.warning("Please select or upload CSV files with a 'Timestamp' column to analyze.")
    st.stop()
