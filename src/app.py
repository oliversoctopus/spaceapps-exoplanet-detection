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
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
        df = pd.read_csv('../data/raw/kepler_koi.csv', comment='#')
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 NASA Exoplanet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Detection of Exoplanets from Kepler Transit Data</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    models = load_models()

    # Sidebar
    with st.sidebar:
        st.image("https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png", width=200)
        st.markdown("## NASA Space Apps Challenge 2025")
        st.markdown("**Challenge:** A World Away: Hunting for Exoplanets with AI")
        st.markdown("---")

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

        st.markdown("---")
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

        st.markdown("---")
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

    # Load sample data for selected model
    X_data, y_data = load_sample_data(model_type)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predictions", "🔍 Feature Explainer", "📊 Data Explorer", "📈 Model Performance", "📚 Documentation"])

    with tab1:
        st.header("Exoplanet Prediction Interface")

        prediction_method = st.radio(
            "Choose prediction method:",
            ["Sample Data", "Upload CSV"],
            help=f"Manual input not available - model uses {len(feature_names)} specialized features"
        )

        if prediction_method == "Sample Data":
            st.subheader("Test with Sample Data")

            if X_data is not None and y_data is not None:
                sample_idx = st.slider("Select sample index", 0, len(X_data)-1, 0, key="prediction_sample_idx")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Sample Features (first 10):**")
                    sample = X_data.iloc[sample_idx]
                    for i, (feat, val) in enumerate(sample.items()):
                        if i < 10:
                            st.write(f"- {feat}: {val:.4f}")

                with col2:
                    if st.button("Classify Sample", type="primary"):
                        sample_input = sample.values.reshape(1, -1)
                        prediction = model.predict(sample_input)[0]
                        probability = model.predict_proba(sample_input)[0]
                        actual = y_data.iloc[sample_idx]

                        st.markdown("**Results:**")
                        if prediction == 1:
                            st.success(f"🌍 Predicted: PLANET (confidence: {probability[1]:.1%})")
                        else:
                            st.warning(f"❌ Predicted: NON-PLANET (confidence: {probability[0]:.1%})")

                        if actual == 1:
                            st.info("Actual: PLANET")
                        else:
                            st.info("Actual: NON-PLANET")

                        if prediction == actual:
                            st.success("✅ Correct prediction!")
                        else:
                            st.error("❌ Incorrect prediction")

                        # Confidence gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability[1] * 100,
                            title={'text': "Planet Likelihood"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)

        else:  # Upload CSV
            st.subheader("Upload Transit Data")
            st.info(f"CSV must contain {len(feature_names)} columns: {', '.join(feature_names[:5])}... and {len(feature_names)-5} more")

            uploaded_file = st.file_uploader(
                "Choose a preprocessed CSV file",
                type="csv"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(df)} samples")

                    # Check for required columns
                    missing_cols = set(feature_names) - set(df.columns)
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                    else:
                        if st.button("Classify All Samples", type="primary"):
                            # Make predictions
                            predictions = model.predict(df[feature_names])
                            probabilities = model.predict_proba(df[feature_names])[:, 1]

                            # Add results to dataframe
                            df['Prediction'] = ['Planet' if p == 1 else 'Non-Planet' for p in predictions]
                            df['Planet_Probability'] = probabilities

                            # Display results
                            st.success(f"Classification complete!")
                            st.write(f"Planets detected: {sum(predictions)} / {len(predictions)}")

                            # Show results table
                            st.dataframe(df[['Prediction', 'Planet_Probability'] + feature_names[:5]])

                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    with tab2:
        st.header("🔍 Feature Importance Explainer")
        st.markdown("""
        This tool uses **SHAP (SHapley Additive exPlanations)** to explain which features
        contribute most to the model's prediction for a specific KOI. Features in **red**
        push the prediction towards "Planet", while features in **blue** push it towards "Non-Planet".
        """)

        if X_data is not None and y_data is not None:
            # Load KOI names if available
            koi_names = load_koi_names()

            if koi_names is not None and len(koi_names) == len(X_data):
                st.subheader("Select a KOI by Name")

                # Create filter options
                col1, col2 = st.columns([2, 1])
                with col1:
                    filter_option = st.selectbox(
                        "Filter KOIs by type:",
                        ["All KOIs", "Confirmed Planets", "Candidate Planets", "False Positives"]
                    )

                # Filter indices based on selection
                if filter_option == "Confirmed Planets" or filter_option == "Candidate Planets":
                    available_indices = [i for i, label in enumerate(y_data) if label == 1]
                elif filter_option == "False Positives":
                    available_indices = [i for i, label in enumerate(y_data) if label == 0]
                else:
                    available_indices = list(range(len(X_data)))

                available_kois = [(koi_names[i], i) for i in available_indices[:100]]  # Limit to first 100

                with col2:
                    st.metric("Available", len(available_kois))

                selected_koi = st.selectbox(
                    "Choose KOI:",
                    available_kois,
                    format_func=lambda x: x[0]
                )

                sample_idx = selected_koi[1]
            else:
                st.subheader("Select a Sample by Index")
                sample_idx = st.slider("Select sample index", 0, len(X_data)-1, 0, key="explainer_sample_idx")
                st.info(f"Sample #{sample_idx}")

            # Display sample information
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Sample Information:**")
                if koi_names is not None and len(koi_names) == len(X_data):
                    st.write(f"**KOI Name:** {koi_names[sample_idx]}")
                actual = y_data.iloc[sample_idx]
                st.write(f"**Actual Label:** {'🌍 Planet' if actual == 1 else '❌ Non-Planet'}")

            with col2:
                # Make prediction
                sample = X_data.iloc[sample_idx].values.reshape(1, -1)
                prediction = model.predict(sample)[0]
                probability = model.predict_proba(sample)[0]

                st.markdown("**Model Prediction:**")
                if prediction == 1:
                    st.success(f"🌍 **Planet** (confidence: {probability[1]:.1%})")
                else:
                    st.warning(f"❌ **Non-Planet** (confidence: {probability[0]:.1%})")

                if prediction == actual:
                    st.success("✅ Correct prediction!")
                else:
                    st.error("❌ Incorrect prediction")

            # Generate SHAP explanation
            if st.button("🔍 Explain This Prediction", type="primary"):
                with st.spinner("Generating SHAP explanation... This may take a moment."):
                    try:
                        # Create background data (sample of training data)
                        background_size = min(100, len(X_data))
                        background_data = X_data.sample(n=background_size, random_state=42)

                        # Get SHAP explainer
                        explainer = get_shap_explainer(model, background_data)

                        # Calculate SHAP values for this sample
                        shap_values = explainer(sample)

                        # Create waterfall plot
                        st.subheader("Feature Contribution Waterfall Plot")
                        st.markdown("""
                        This plot shows how each feature pushes the prediction from the base value (average model output)
                        to the final prediction. The base value represents what the model would predict on average.
                        """)

                        fig = create_shap_waterfall_plot(shap_values[0], feature_names, max_display=15)
                        st.pyplot(fig)
                        plt.close()

                        # Show top contributing features
                        st.subheader("Top Contributing Features")

                        # Get feature importance for this prediction
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP Value': shap_values.values[0],
                            'Feature Value': sample[0]
                        })
                        feature_importance['Absolute Impact'] = abs(feature_importance['SHAP Value'])
                        feature_importance = feature_importance.sort_values('Absolute Impact', ascending=False)

                        # Display top 10 features
                        st.dataframe(
                            feature_importance[['Feature', 'Feature Value', 'SHAP Value']].head(10).style.format({
                                'Feature Value': '{:.4f}',
                                'SHAP Value': '{:.4f}'
                            }),
                            use_container_width=True
                        )

                        st.markdown("""
                        **How to interpret:**
                        - **Positive SHAP values** (push towards Planet): These features increase the likelihood of a planet classification
                        - **Negative SHAP values** (push towards Non-Planet): These features decrease the likelihood of a planet classification
                        - **Magnitude**: Larger absolute values indicate stronger influence on the prediction
                        """)

                    except Exception as e:
                        st.error(f"Error generating SHAP explanation: {str(e)}")
                        st.info("SHAP analysis may fail for some models. Try selecting a different sample.")
        else:
            st.warning("Sample data not available for feature explanation.")

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

if __name__ == "__main__":
    main()
