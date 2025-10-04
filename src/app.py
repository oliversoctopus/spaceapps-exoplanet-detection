"""
NASA Exoplanet Detection - Interactive Streamlit Application
============================================================
Interactive web app for detecting exoplanets using Kepler transit data.
Created for NASA Space Apps Challenge 2025.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="NASA Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-top: 0rem;
        margin-bottom: 0.5rem;
        padding-top: 0rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Reduce top padding of main content area */
    .block-container {
        padding-top: 1rem;
    }
    /* Reduce horizontal rule margins */
    hr {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load both baseline and full LightGBM models"""
    models = {}

    # Load baseline model
    try:
        with open('../models/baseline_lightgbm_model.pkl', 'rb') as f:
            models['baseline_model'] = pickle.load(f)
        with open('../models/baseline_model_metrics.json', 'r') as f:
            models['baseline_metrics'] = json.load(f)
        with open('../data/preprocessing/baseline_preprocessor.pkl', 'rb') as f:
            models['baseline_preprocessor'] = pickle.load(f)
    except Exception as e:
        st.warning(f"Baseline model not available: {str(e)}")
        models['baseline_model'] = None

    # Load full model
    try:
        with open('../models/lightgbm_model.pkl', 'rb') as f:
            models['full_model'] = pickle.load(f)
        with open('../models/model_metrics.json', 'r') as f:
            models['full_metrics'] = json.load(f)
        with open('../data/preprocessing/preprocessor.pkl', 'rb') as f:
            models['full_preprocessor'] = pickle.load(f)
    except Exception as e:
        st.warning(f"Full model not available: {str(e)}")
        models['full_model'] = None

    return models

@st.cache_data
def load_sample_data(model_type):
    """Load preprocessed data for demonstration"""
    try:
        if model_type == "baseline":
            df = pd.read_csv('../data/preprocessing/kepler_koi_baseline.csv')
        else:
            df = pd.read_csv('../data/preprocessing/kepler_koi_preprocessed.csv')
        X = df.drop('label', axis=1)
        y = df['label']
        return X, y
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None

@st.cache_data
def load_koi_names():
    """Load original KOI names for sample selection"""
    try:
        df = pd.read_csv('../data/kepler_koi.csv', comment='#')
        # Get KOI names that exist in our preprocessed data
        df_processed = pd.read_csv('../data/preprocessing/kepler_koi_preprocessed.csv')
        return df['kepoi_name'].iloc[:len(df_processed)].tolist()
    except Exception as e:
        return None

@st.cache_resource
def get_shap_explainer(_model, background_data):
    """Create SHAP explainer (cached to avoid recomputation)"""
    # Use a small background dataset for TreeExplainer
    return shap.TreeExplainer(_model, background_data)

def create_shap_waterfall_plot(shap_values, feature_names, max_display=10):
    """Create SHAP waterfall plot using matplotlib and convert to Streamlit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    return fig

@st.cache_data
def load_raw_planet_data():
    """Load planet radius data from raw dataset"""
    try:
        df_raw = pd.read_csv('../data/kepler_koi.csv', comment='#')
        # Create a mapping of indices to planet radius
        radius_data = {}
        for idx, row in df_raw.iterrows():
            if pd.notna(row.get('koi_prad')):
                radius_data[idx] = float(row['koi_prad'])
        return radius_data
    except Exception as e:
        st.warning(f"Could not load raw planet data: {str(e)}")
        return {}

def get_planet_radius(koi_idx, radius_data):
    """Get planet radius from cached raw data"""
    return radius_data.get(koi_idx, None)

def create_planet_comparison_3d(planet_radius, planet_type="Exoplanet"):
    """Create 3D visualization comparing planet size to Earth using Plotly"""

    # Planet colors based on size classification
    if planet_radius is None:
        return None

    if planet_radius < 1.5:
        planet_color = '#9F2B00'  # Terrestrial (Rust)
        type_name = "Terrestrial"
    elif planet_radius < 4:
        planet_color = '#AF4425'  # Super Earth (Burnt Sienna)
        type_name = "Super Earth"
    elif planet_radius < 10:
        planet_color = '#5B92E5'  # Neptune-like (Cornflower Blue)
        type_name = "Neptune-like"
    else:
        planet_color = '#D2B48C'  # Gas Giant (Tan)
        type_name = "Gas Giant"

    earth_color = '#4F7CAC'  # Earth (Steel Blue)

    # Create spheres with appropriate scaling
    max_display_radius = 3
    earth_display_radius = 0.5
    planet_display_radius = min(max_display_radius, earth_display_radius * planet_radius)

    # Create mesh grids for spheres
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    # Planet sphere
    x_planet = planet_display_radius * np.outer(np.cos(u), np.sin(v)) - (planet_display_radius + 0.5)
    y_planet = planet_display_radius * np.outer(np.sin(u), np.sin(v))
    z_planet = planet_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Earth sphere
    x_earth = earth_display_radius * np.outer(np.cos(u), np.sin(v)) + (earth_display_radius + 0.5)
    y_earth = earth_display_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create 3D plot
    fig = go.Figure()

    # Add planet
    fig.add_trace(go.Surface(
        x=x_planet, y=y_planet, z=z_planet,
        colorscale=[[0, planet_color], [1, planet_color]],
        showscale=False,
        name=f'{type_name} Planet',
        hovertemplate=f'<b>{type_name}</b><br>Radius: {planet_radius:.2f}x Earth<extra></extra>'
    ))

    # Add Earth
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, earth_color], [1, earth_color]],
        showscale=False,
        name='Earth',
        hovertemplate='<b>Earth</b><br>Radius: 1.00x Earth<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f'Size Comparison: {type_name} ({planet_radius:.2f}x Earth) vs Earth',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#111827',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor='#1F2937',
        plot_bgcolor='#1F2937',
        font=dict(color='#F3F4F6'),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig, type_name

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 NASA Exoplanet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Detection of Exoplanets from Kepler Transit Data</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    models = load_models()

    # Sidebar
    with st.sidebar:
        # Model selection
        st.markdown("### Select Model")
        available_models = []
        if models.get('baseline_model') is not None:
            available_models.append("Baseline (9 features)")
        if models.get('full_model') is not None:
            available_models.append("Full (52 features)")

        if not available_models:
            st.error("No models available!")
            return

        selected_model = st.radio(
            "Choose model:",
            available_models,
            help="Baseline model uses 9 core features, Full model uses 52 features (optimized)"
        )

        # Set active model based on selection
        if "Baseline" in selected_model:
            model = models['baseline_model']
            metrics = models['baseline_metrics']
            preprocessor = models['baseline_preprocessor']
            model_type = "baseline"
        else:
            model = models['full_model']
            metrics = models['full_metrics']
            preprocessor = models['full_preprocessor']
            model_type = "full"

        feature_names = preprocessor['feature_columns']

    # Load sample data for selected model
    X_data, y_data = load_sample_data(model_type)

    # Load planet radius data from raw dataset
    radius_data = load_raw_planet_data()

    # Load KOI names if available
    koi_names = load_koi_names()

    # Continue sidebar with Sample Explorer controls
    with st.sidebar:
        # Only show sample selection if in Sample Explorer tab and data is available
        if X_data is not None and y_data is not None:
            st.markdown("### 🔍 Sample Selection")

            # Filter options
            filter_option = st.radio(
                "Filter by type:",
                ["Planets", "False Positives"],
                key="tab1_filter"
            )

            # Filter indices based on selection
            if filter_option == "Planets":
                available_indices = [i for i, label in enumerate(y_data) if label == 1]
            else:  # False Positives
                available_indices = [i for i, label in enumerate(y_data) if label == 0]

            st.metric("Available Samples", len(available_indices))

            # Slider for selection
            if len(available_indices) > 0:
                slider_idx = st.slider(
                    "Select sample:",
                    0,
                    len(available_indices) - 1,
                    0,
                    key="tab1_slider"
                )
                sample_idx = available_indices[slider_idx]

                # Display KOI name if available
                if koi_names is not None and len(koi_names) == len(X_data):
                    st.info(f"**KOI:** {koi_names[sample_idx]}")
                else:
                    st.info(f"**Sample Index:** {sample_idx}")

                # Navigation section
                st.markdown("### 📑 View Section")
                view_section = st.radio(
                    "Jump to:",
                    ["Prediction", "3D Visualization", "Feature Explanation"],
                    key="tab1_section"
                )
            else:
                st.warning("No samples available for this filter.")
                sample_idx = 0
                view_section = "Prediction"
        else:
            sample_idx = 0
            view_section = "Prediction"

        # Model performance and about section
        st.markdown("### Model Performance")

        # Display test metrics
        test_metrics = metrics['metrics']['test']

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{test_metrics['accuracy']:.1%}")
            st.metric("Precision", f"{test_metrics['precision']:.1%}")
        with col2:
            st.metric("Recall", f"{test_metrics['recall']:.1%}")
            st.metric("F1-Score", f"{test_metrics['f1']:.1%}")

        st.metric("ROC-AUC", f"{test_metrics['roc_auc']:.1%}")

        st.markdown("### About")
        st.markdown(f"""
        This application uses **LightGBM** to classify
        potential exoplanets from Kepler Space Telescope data.

        **Model:** {selected_model}
        **Features:** {len(feature_names)} observational features
        **Accuracy:** {test_metrics['accuracy']:.1%}

        Data leakage has been carefully removed to ensure
        the model learns from genuine transit signals only.
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Sample Explorer", "📂 Import & Predict", "📊 Data Explorer", "📈 Model Performance", "📚 Documentation"])

    with tab1:
        st.header("🔍 Sample Explorer & Analysis")

        if X_data is not None and y_data is not None:

            # Make prediction
            sample = X_data.iloc[sample_idx].values.reshape(1, -1)
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0]
            actual = y_data.iloc[sample_idx]

            # Section 1: Prediction Results
            if view_section == "Prediction" or view_section == "All":
                st.subheader("📊 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**KOI Information:**")
                    if koi_names is not None and len(koi_names) == len(X_data):
                        st.write(f"**Name:** {koi_names[sample_idx]}")
                    st.write(f"**Index:** {sample_idx}")
                    st.write(f"**Actual Label:** {'🌍 Planet' if actual == 1 else '❌ Non-Planet'}")

                with col2:
                    st.markdown("**Model Prediction:**")
                    if prediction == 1:
                        st.success(f"🌍 **Planet**")
                        st.write(f"Confidence: {probability[1]:.1%}")
                    else:
                        st.warning(f"❌ **Non-Planet**")
                        st.write(f"Confidence: {probability[0]:.1%}")

                with col3:
                    st.markdown("**Classification:**")
                    if prediction == actual:
                        st.success("✅ Correct")
                    else:
                        st.error("❌ Incorrect")

                # Confidence gauge
                st.markdown("**Prediction Confidence:**")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    title={'text': "Planet Likelihood (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#60a5fa" if prediction == 1 else "#f87171"},
                        'steps': [
                            {'range': [0, 50], 'color': "#1f2937"},
                            {'range': [50, 100], 'color': "#374151"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=300,
                    paper_bgcolor='#1F2937',
                    font=dict(color='#F3F4F6')
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Section 2: 3D Visualization
            elif view_section == "3D Visualization":
                st.subheader("🪐 Planet Size Comparison")

                planet_radius = get_planet_radius(sample_idx, radius_data)
                if planet_radius is not None and planet_radius > 0:
                    viz_result = create_planet_comparison_3d(planet_radius)
                    if viz_result:
                        fig_3d, planet_type = viz_result
                        st.plotly_chart(fig_3d, use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Planet Type", planet_type)
                        with col2:
                            st.metric("Radius", f"{planet_radius:.2f}× Earth")
                        with col3:
                            if planet_radius < 1.5:
                                st.info("Similar to Earth/Mars")
                            elif planet_radius < 4:
                                st.info("Larger than Earth")
                            elif planet_radius < 10:
                                st.info("Similar to Neptune")
                            else:
                                st.info("Similar to Jupiter")
                    else:
                        st.warning("Unable to create 3D visualization for this planet.")
                else:
                    st.info("Planet radius data not available for this KOI.")
                    st.markdown("""
                    **Note:** Radius data is only available for confirmed planets and candidates.
                    False positives typically don't have reliable radius measurements.
                    """)

            # Section 3: Feature Explanation
            elif view_section == "Feature Explanation":
                st.subheader("🔍 Feature Importance Analysis")
                st.markdown("""
                This analysis uses **SHAP (SHapley Additive exPlanations)** to show which features
                contributed most to the model's prediction for this specific KOI.
                """)

                if st.button("🔍 Generate Explanation", type="primary", key="tab1_explain"):
                    with st.spinner("Generating SHAP explanation... This may take a moment."):
                        try:
                            # Create background data
                            background_size = min(100, len(X_data))
                            background_data = X_data.sample(n=background_size, random_state=42)

                            # Get SHAP explainer
                            explainer = get_shap_explainer(model, background_data)

                            # Calculate SHAP values
                            shap_values = explainer(sample)

                            # Create waterfall plot
                            st.markdown("**Feature Contribution Waterfall:**")
                            fig_shap = create_shap_waterfall_plot(shap_values[0], feature_names, max_display=15)
                            st.pyplot(fig_shap)
                            plt.close()

                            # Show top contributing features
                            st.markdown("**Top Contributing Features:**")

                            feature_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'SHAP Value': shap_values.values[0],
                                'Feature Value': sample[0]
                            })
                            feature_importance['Absolute Impact'] = abs(feature_importance['SHAP Value'])
                            feature_importance = feature_importance.sort_values('Absolute Impact', ascending=False)

                            st.dataframe(
                                feature_importance[['Feature', 'Feature Value', 'SHAP Value']].head(10).style.format({
                                    'Feature Value': '{:.4f}',
                                    'SHAP Value': '{:.4f}'
                                }),
                                use_container_width=True
                            )

                            st.markdown("""
                            **How to interpret:**
                            - **Positive SHAP values** (red bars): Push prediction towards Planet
                            - **Negative SHAP values** (blue bars): Push prediction towards Non-Planet
                            - **Magnitude**: Larger absolute values = stronger influence on the prediction
                            """)

                        except Exception as e:
                            st.error(f"Error generating SHAP explanation: {str(e)}")
        else:
            st.warning("Sample data not available.")

    with tab2:
        st.header("📂 Import & Predict on CSV Data")
        st.markdown("""
        Upload your own preprocessed exoplanet transit data to get predictions from the model.
        The CSV must contain all required features with proper preprocessing.
        """)

        st.info(f"**Required:** CSV must contain {len(feature_names)} columns: {', '.join(feature_names[:5])}... and {len(feature_names)-5} more")

        uploaded_file = st.file_uploader(
            "Choose a preprocessed CSV file",
            type="csv",
            key="tab2_upload"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} samples")

                # Check for required columns
                missing_cols = set(feature_names) - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(list(missing_cols)[:10])}")
                else:
                    if st.button("🔮 Classify All Samples", type="primary", key="tab2_classify"):
                        with st.spinner("Making predictions..."):
                            # Make predictions
                            predictions = model.predict(df[feature_names])
                            probabilities = model.predict_proba(df[feature_names])[:, 1]

                            # Add results to dataframe
                            df['Prediction'] = ['Planet' if p == 1 else 'Non-Planet' for p in predictions]
                            df['Planet_Probability'] = probabilities

                            # Display results
                            st.success(f"✅ Classification complete!")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Planets Detected", sum(predictions))
                            with col2:
                                st.metric("Non-Planets Detected", len(predictions) - sum(predictions))

                            # Show results table
                            st.subheader("Prediction Results")
                            st.dataframe(
                                df[['Prediction', 'Planet_Probability'] + feature_names[:5]].head(50),
                                use_container_width=True
                            )

                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    with tab3:
        st.header("Data Explorer")

        if X_data is not None:
            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(X_data))
            with col2:
                planet_count = sum(y_data == 1)
                st.metric("Planets", planet_count)
            with col3:
                st.metric("Non-Planets", len(y_data) - planet_count)

            # Feature statistics
            st.subheader("Feature Statistics (first 10 features)")
            stats = X_data.iloc[:, :10].describe()
            st.dataframe(stats.style.format("{:.4f}"))

            # Feature distributions
            st.subheader("Feature Distributions (first 4 features)")
            fig = make_subplots(rows=2, cols=2, subplot_titles=X_data.columns[:4].tolist())

            for idx, col in enumerate(X_data.columns[:4]):
                row = idx // 2 + 1
                col_pos = idx % 2 + 1
                fig.add_trace(
                    go.Histogram(x=X_data[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )

            fig.update_layout(height=600, title_text="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Model Performance Analysis")

        # Metrics comparison across splits
        st.subheader("Performance Across Data Splits")

        splits_df = pd.DataFrame({
            'Split': ['Train', 'Validation', 'Test'],
            'Accuracy': [metrics['metrics'][s]['accuracy'] for s in ['train', 'validation', 'test']],
            'F1-Score': [metrics['metrics'][s]['f1'] for s in ['train', 'validation', 'test']],
            'ROC-AUC': [metrics['metrics'][s]['roc_auc'] for s in ['train', 'validation', 'test']]
        })

        fig_splits = px.line(splits_df, x='Split', y=['Accuracy', 'F1-Score', 'ROC-AUC'],
                            title='Model Performance Across Splits',
                            markers=True)
        fig_splits.update_layout(height=400, yaxis_title="Score", yaxis_range=[0.8, 1.0])
        st.plotly_chart(fig_splits, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Test Set Metrics")
            test_metrics = metrics['metrics']['test']
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [
                    test_metrics['accuracy'],
                    test_metrics['precision'],
                    test_metrics['recall'],
                    test_metrics['f1'],
                    test_metrics['roc_auc']
                ]
            })

            fig_metrics = px.bar(
                metrics_df, x='Metric', y='Score',
                title='Test Set Performance',
                color='Score',
                color_continuous_scale='viridis'
            )
            fig_metrics.update_layout(height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig_metrics, use_container_width=True)

        with col2:
            st.subheader("Model Information")
            st.json(metrics['model_params'])

    with tab5:
        st.header("Documentation")

        model_description = "Baseline (9 features)" if model_type == "baseline" else "Full (52 features, optimized)"

        st.markdown(f"""
        ## About This Project

        This project was developed for the **NASA Space Apps Challenge 2025**, specifically for the
        "A World Away: Hunting for Exoplanets with AI" challenge.

        ### Dataset
        - **Source**: NASA Exoplanet Archive - Kepler Objects of Interest (KOI)
        - **Size**: 9,564 transit signals
        - **Classes**: Planets (CONFIRMED + CANDIDATE) vs Non-Planets (FALSE POSITIVE)
        - **Current Model**: {model_description}

        ### Model Architecture
        - **Algorithm**: LightGBM (Light Gradient Boosting Machine)
        - **Type**: Binary classification (Planet vs Non-Planet)
        - **Data Split**: 70% train, 10% validation, 20% test
        - **Features**: {len(feature_names)} features

        ### Available Models

        **Baseline Model (9 features):**
        - Core stellar properties: koi_steff, koi_slogg, koi_srad, koi_kepmag
        - Core transit signals: koi_period, koi_depth, koi_duration, koi_impact, koi_model_snr
        - Test Accuracy: ~84%

        **Full Model (52 features, optimized):**
        - Comprehensive stellar properties (temperature, radius, mass, gravity, metallicity)
        - Transit signals (period, depth, duration, impact)
        - Photometry (magnitudes in multiple bands)
        - Signal statistics (SNR, evidence)
        - Optimized with aggressive regularization to reduce overfitting
        - Test Accuracy: ~87%, ROC-AUC: ~94%

        ### Data Leakage Prevention
        We removed 88 columns that could leak classification information:
        - False positive flags (koi_fpflag_*)
        - Derived planet properties (koi_teq, koi_dor)
        - Disposition scores and vetting metadata
        - Model fitting outputs
        - Centroid offset features (dikco_*, dicco_*, fwm_*)

        The model uses only genuine observational data: stellar properties,
        photometry, and transit signal measurements.

        ### Performance (Test Set)
        - **Accuracy**: {test_metrics['accuracy']:.1%}
        - **Precision**: {test_metrics['precision']:.1%}
        - **Recall**: {test_metrics['recall']:.1%}
        - **F1-Score**: {test_metrics['f1']:.1%}
        - **ROC-AUC**: {test_metrics['roc_auc']:.1%}

        ### How to Use
        1. **Sample Data**: Test with samples from the preprocessed dataset
        2. **Upload CSV**: Upload preprocessed transit data with {len(feature_names)} features

        ### Team
        - Developed for NASA Space Apps Challenge 2025
        - Challenge: A World Away - Hunting for Exoplanets with AI

        ### References
        - NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
        - Kepler Mission: https://www.nasa.gov/mission_pages/kepler/
        - LightGBM: https://lightgbm.readthedocs.io/

        ### Citation
        ```
        NASA Exoplanet Archive (2025)
        DOI: http://doi.org/10.17616/R3X31K
        ```
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #9CA3AF; padding: 20px;'>
        <p><strong>NASA Space Apps Challenge 2025</strong></p>
        <p>Challenge: A World Away - Hunting for Exoplanets with AI</p>
        <p style='font-size: 0.9em;'>Data from NASA Exoplanet Archive | Powered by LightGBM & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
