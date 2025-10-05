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
import streamlit.components.v1 as components

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

@st.cache_data
def load_candidate_kois(model_type):
    """Load KOIs marked as CANDIDATE (unconfirmed) for prediction"""
    try:
        # Load raw data
        df_raw = pd.read_csv('../data/kepler_koi.csv', comment='#')

        # Filter for candidates only
        candidates_df = df_raw[df_raw['koi_disposition'] == 'CANDIDATE'].copy()

        if len(candidates_df) == 0:
            return None, None, None

        # Load preprocessor
        if model_type == "baseline":
            with open('../data/preprocessing/baseline_preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
        else:
            with open('../data/preprocessing/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)

        feature_cols = preprocessor['feature_columns']

        # Check if all required features exist
        missing_cols = set(feature_cols) - set(candidates_df.columns)
        if missing_cols:
            st.warning(f"Some features missing from raw data: {len(missing_cols)} columns")
            return None, None, None

        # Extract features
        X_candidates = candidates_df[feature_cols].copy()

        # Apply same preprocessing (imputation and scaling)
        imputer = preprocessor['imputer']
        scaler = preprocessor['scaler']

        X_candidates_processed = imputer.transform(X_candidates)
        X_candidates_processed = scaler.transform(X_candidates_processed)
        X_candidates_processed = pd.DataFrame(X_candidates_processed, columns=feature_cols)

        # Store metadata
        metadata = candidates_df[['kepoi_name', 'koi_period', 'koi_depth', 'koi_prad', 'koi_steff']].copy()
        metadata.reset_index(drop=True, inplace=True)

        return X_candidates_processed, metadata, candidates_df.index.tolist()

    except Exception as e:
        st.error(f"Error loading candidates: {str(e)}")
        return None, None, None

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
    """Load complete system data from raw dataset"""
    try:
        df_raw = pd.read_csv('../data/kepler_koi.csv', comment='#')
        # Create a mapping of indices to system data
        system_data = {}
        for idx, row in df_raw.iterrows():
            system_data[idx] = {
                'planet_radius': row.get('koi_prad'),  # Earth radii
                'star_radius': row.get('koi_srad'),    # Solar radii
                'orbital_distance': row.get('koi_sma'), # AU
                'orbital_period': row.get('koi_period'), # days
                'star_mass': row.get('koi_smass'),      # Solar masses
                'star_temp': row.get('koi_steff')       # Kelvin
            }
        return system_data
    except Exception as e:
        st.warning(f"Could not load raw planet data: {str(e)}")
        return {}

def get_system_data(koi_idx, system_data):
    """Get complete system data for a KOI"""
    return system_data.get(koi_idx, {})

def classify_star_by_temp(temp_kelvin):
    """Classify star type by temperature and return color"""
    if pd.isna(temp_kelvin):
        return 'G-type', '#FDB813'  # Default to Sun-like

    if temp_kelvin >= 30000:
        return 'O-type', '#9BB0FF'  # Blue
    elif temp_kelvin >= 10000:
        return 'B-type', '#AABFFF'  # Blue-white
    elif temp_kelvin >= 7500:
        return 'A-type', '#CAD7FF'  # White
    elif temp_kelvin >= 6000:
        return 'F-type', '#F8F7FF'  # Yellow-white
    elif temp_kelvin >= 5200:
        return 'G-type', '#FDB813'  # Yellow (Sun-like)
    elif temp_kelvin >= 3700:
        return 'K-type', '#FF8912'  # Orange
    else:
        return 'M-type', '#FF4500'  # Red

def create_threejs_visualization(system_data, view_mode='system'):
    """Create Three.js HTML visualization for solar system"""

    planet_radius = system_data.get('planet_radius', 1.0)
    star_radius = system_data.get('star_radius', 1.0)
    star_temp = system_data.get('star_temp', 5778)

    # Handle missing data
    if pd.isna(planet_radius):
        planet_radius = 1.0
    if pd.isna(star_radius):
        star_radius = 1.0
    if pd.isna(star_temp):
        star_temp = 5778

    # Determine planet color based on size
    if planet_radius < 1.5:
        planet_color = '0x9F2B00'  # Terrestrial
    elif planet_radius < 4:
        planet_color = '0xAF4425'  # Super Earth
    elif planet_radius < 10:
        planet_color = '0x5B92E5'  # Neptune-like
    else:
        planet_color = '0xD2B48C'  # Gas Giant

    # Different view modes
    if view_mode == 'star':
        # Star comparison view
        scene_setup = f"""
            // Star comparison: Host star vs Sun
            const starProps = getStarProperties({star_temp});

            // Host star (left)
            const hostStarRadius = {star_radius} * 20;
            const hostStarGeometry = new THREE.SphereGeometry(hostStarRadius, 64, 64);
            const hostStarMaterial = new THREE.MeshBasicMaterial({{
                color: starProps.color
            }});
            const hostStarMesh = new THREE.Mesh(hostStarGeometry, hostStarMaterial);

            // Host star glow
            const hostGlowGeometry = new THREE.SphereGeometry(hostStarRadius * 1.15, 64, 64);
            const hostGlowMaterial = new THREE.MeshBasicMaterial({{
                color: starProps.color,
                transparent: true,
                opacity: 0.3
            }});
            const hostGlow = new THREE.Mesh(hostGlowGeometry, hostGlowMaterial);
            hostStarMesh.add(hostGlow);

            // Sun (right, for comparison)
            const sunRadius = 1.0 * 20;  // Same scale
            const sunGeometry = new THREE.SphereGeometry(sunRadius, 64, 64);
            const sunMaterial = new THREE.MeshBasicMaterial({{
                color: 0xFDB813
            }});
            const sunMesh = new THREE.Mesh(sunGeometry, sunMaterial);

            // Sun glow
            const sunGlowGeometry = new THREE.SphereGeometry(sunRadius * 1.15, 64, 64);
            const sunGlowMaterial = new THREE.MeshBasicMaterial({{
                color: 0xFDB813,
                transparent: true,
                opacity: 0.3
            }});
            const sunGlow = new THREE.Mesh(sunGlowGeometry, sunGlowMaterial);
            sunMesh.add(sunGlow);

            // Position stars side by side
            const spacing = Math.max(hostStarRadius, sunRadius) * 3;
            hostStarMesh.position.x = -spacing / 2;
            sunMesh.position.x = spacing / 2;

            scene.add(hostStarMesh);
            scene.add(sunMesh);

            // Add text labels
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 512;
            canvas.height = 128;

            // Host star label
            ctx.fillStyle = 'white';
            ctx.font = 'bold 48px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Host Star', 256, 64);
            const hostLabelTexture = new THREE.CanvasTexture(canvas);
            const hostLabelMaterial = new THREE.SpriteMaterial({{ map: hostLabelTexture }});
            const hostLabel = new THREE.Sprite(hostLabelMaterial);
            hostLabel.position.set(-spacing / 2, -hostStarRadius - spacing * 0.2, 0);
            hostLabel.scale.set(spacing * 0.4, spacing * 0.1, 1);
            scene.add(hostLabel);

            // Sun label
            const canvas2 = document.createElement('canvas');
            const ctx2 = canvas2.getContext('2d');
            canvas2.width = 512;
            canvas2.height = 128;
            ctx2.fillStyle = 'white';
            ctx2.font = 'bold 48px Arial';
            ctx2.textAlign = 'center';
            ctx2.fillText('Sun (Reference)', 256, 64);
            const sunLabelTexture = new THREE.CanvasTexture(canvas2);
            const sunLabelMaterial = new THREE.SpriteMaterial({{ map: sunLabelTexture }});
            const sunLabel = new THREE.Sprite(sunLabelMaterial);
            sunLabel.position.set(spacing / 2, -sunRadius - spacing * 0.2, 0);
            sunLabel.scale.set(spacing * 0.4, spacing * 0.1, 1);
            scene.add(sunLabel);

            // Position camera
            camera.position.set(0, spacing * 0.3, spacing * 1.2);
            camera.lookAt(0, 0, 0);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 1);
            scene.add(ambientLight);

            // Animation
            const clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const delta = clock.getDelta();
                controls.update();

                hostStarMesh.rotation.y += delta * 0.1;
                sunMesh.rotation.y += delta * 0.1;

                renderer.render(scene, camera);
            }}
        """
    elif view_mode == 'planet':
        # Planet comparison view
        scene_setup = f"""
            // Planet comparison: Exoplanet vs Earth

            // Exoplanet (left)
            const exoPlanetRadius = {planet_radius} * 2;
            const exoPlanetGeometry = new THREE.SphereGeometry(exoPlanetRadius, 64, 64);
            const exoPlanetMaterial = new THREE.MeshStandardMaterial({{
                color: {planet_color},
                roughness: 0.6,
                metalness: 0.1
            }});
            const exoPlanetMesh = new THREE.Mesh(exoPlanetGeometry, exoPlanetMaterial);

            // Earth (right, for comparison)
            const earthRadius = 1.0 * 2;  // Same scale
            const earthGeometry = new THREE.SphereGeometry(earthRadius, 64, 64);
            const earthMaterial = new THREE.MeshStandardMaterial({{
                color: 0x4F7CAC,
                roughness: 0.6,
                metalness: 0.1
            }});
            const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);

            // Position planets side by side
            const spacing = Math.max(exoPlanetRadius, earthRadius) * 3.5;
            exoPlanetMesh.position.x = -spacing / 2;
            earthMesh.position.x = spacing / 2;

            scene.add(exoPlanetMesh);
            scene.add(earthMesh);

            // Add text labels
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 512;
            canvas.height = 128;

            // Exoplanet label
            ctx.fillStyle = 'white';
            ctx.font = 'bold 48px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Exoplanet', 256, 64);
            const exoLabelTexture = new THREE.CanvasTexture(canvas);
            const exoLabelMaterial = new THREE.SpriteMaterial({{ map: exoLabelTexture }});
            const exoLabel = new THREE.Sprite(exoLabelMaterial);
            exoLabel.position.set(-spacing / 2, -exoPlanetRadius - spacing * 0.15, 0);
            exoLabel.scale.set(spacing * 0.35, spacing * 0.09, 1);
            scene.add(exoLabel);

            // Earth label
            const canvas2 = document.createElement('canvas');
            const ctx2 = canvas2.getContext('2d');
            canvas2.width = 512;
            canvas2.height = 128;
            ctx2.fillStyle = 'white';
            ctx2.font = 'bold 48px Arial';
            ctx2.textAlign = 'center';
            ctx2.fillText('Earth (Reference)', 256, 64);
            const earthLabelTexture = new THREE.CanvasTexture(canvas2);
            const earthLabelMaterial = new THREE.SpriteMaterial({{ map: earthLabelTexture }});
            const earthLabel = new THREE.Sprite(earthLabelMaterial);
            earthLabel.position.set(spacing / 2, -earthRadius - spacing * 0.15, 0);
            earthLabel.scale.set(spacing * 0.35, spacing * 0.09, 1);
            scene.add(earthLabel);

            // Position camera
            camera.position.set(0, spacing * 0.3, spacing * 1.2);
            camera.lookAt(0, 0, 0);

            // Simple, clean lighting from one direction
            const mainLight = new THREE.DirectionalLight(0xffffff, 1.8);
            mainLight.position.set(5, 3, 8);
            scene.add(mainLight);

            const ambientLight = new THREE.AmbientLight(0x606060, 0.6);
            scene.add(ambientLight);

            // Animation
            const clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const delta = clock.getDelta();
                controls.update();

                exoPlanetMesh.rotation.y += delta * 0.2;
                earthMesh.rotation.y += delta * 0.2;

                renderer.render(scene, camera);
            }}
        """
    else:  # system view
        scene_setup = f"""
            // Full solar system view
            const starProps = getStarProperties({star_temp});

            // Create planet with size capping to prevent overlap
            let planetRadius = Math.max({planet_radius}, 0.5);

            // Cap planet radius to prevent it from being larger than the star
            const starRadius = {star_radius} * 15;
            const maxPlanetRadius = starRadius * 0.6;
            if (planetRadius > maxPlanetRadius) {{
                planetRadius = maxPlanetRadius;
            }}

            // Create star
            const starGeometry = new THREE.SphereGeometry(starRadius, 64, 64);
            const starMaterial = new THREE.MeshBasicMaterial({{
                color: starProps.color,
                emissive: starProps.color,
                emissiveIntensity: 0.8
            }});
            const starMesh = new THREE.Mesh(starGeometry, starMaterial);

            // Star glow
            const starGlowGeometry = new THREE.SphereGeometry(starRadius * 1.2, 64, 64);
            const starGlowMaterial = new THREE.MeshBasicMaterial({{
                color: starProps.color,
                transparent: true,
                opacity: 0.3
            }});
            const starGlow = new THREE.Mesh(starGlowGeometry, starGlowMaterial);
            starMesh.add(starGlow);

            // Create planet
            const planetGeometry = new THREE.SphereGeometry(planetRadius, 32, 32);
            const planetMaterial = new THREE.MeshStandardMaterial({{
                color: {planet_color},
                roughness: 0.8,
                metalness: 0.1
            }});
            const planetMesh = new THREE.Mesh(planetGeometry, planetMaterial);

            // Position objects - ensure enough separation
            const minSeparation = (starRadius + planetRadius) * 1.5;
            const separation = Math.max(starRadius + 30, minSeparation);
            starMesh.position.x = -separation / 2;
            planetMesh.position.x = separation / 2;

            scene.add(starMesh);
            scene.add(planetMesh);

            // Lighting
            const mainLight = new THREE.PointLight(0xffffff, starProps.intensity * 800, 3000);
            mainLight.position.copy(starMesh.position);
            scene.add(mainLight);

            const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
            scene.add(ambientLight);

            // Create orbital path
            const orbitRadius = separation;
            const orbitGeometry = new THREE.BufferGeometry();
            const orbitPoints = [];
            for (let i = 0; i <= 100; i++) {{
                const angle = (i / 100) * Math.PI * 2;
                orbitPoints.push(
                    starMesh.position.x + Math.cos(angle) * orbitRadius,
                    0,
                    Math.sin(angle) * orbitRadius
                );
            }}
            orbitGeometry.setAttribute('position', new THREE.Float32BufferAttribute(orbitPoints, 3));
            const orbitMaterial = new THREE.LineBasicMaterial({{
                color: 0x888888,
                transparent: true,
                opacity: 0.5
            }});
            const orbitLine = new THREE.Line(orbitGeometry, orbitMaterial);
            scene.add(orbitLine);

            // Position camera
            camera.position.set(50, 30, 80);
            camera.lookAt(0, 0, 0);

            // Animation
            const clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const delta = clock.getDelta();
                controls.update();

                starMesh.rotation.y += delta * 0.1;
                planetMesh.rotation.y += delta * 0.3;

                const time = clock.getElapsedTime() * 0.2;
                planetMesh.position.x = starMesh.position.x + Math.cos(time) * orbitRadius;
                planetMesh.position.z = Math.sin(time) * orbitRadius;

                renderer.render(scene, camera);
            }}
        """

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background: #0a0a0a; }}
            #container {{ width: 100%; height: 600px; background: #000; }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
            }}
        }}
        </script>
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            console.log('Three.js script starting...');

            const container = document.getElementById('container');
            console.log('Container dimensions:', container.clientWidth, container.clientHeight);

            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);

            // Camera setup
            const camera = new THREE.PerspectiveCamera(
                75,
                container.clientWidth / container.clientHeight,
                0.1,
                2000
            );
            camera.position.set(50, 30, 80);
            camera.lookAt(0, 0, 0);

            // Renderer setup
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // Controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Starfield background
            const starVertices = [];
            for (let i = 0; i < 10000; i++) {{
                const x = (Math.random() - 0.5) * 2000;
                const y = (Math.random() - 0.5) * 2000;
                const z = (Math.random() - 0.5) * 2000;
                starVertices.push(x, y, z);
            }}
            const starfieldGeometry = new THREE.BufferGeometry();
            starfieldGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
            const starfieldMaterial = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.5 }});
            const stars = new THREE.Points(starfieldGeometry, starfieldMaterial);
            scene.add(stars);

            // Get star properties based on temperature
            function getStarProperties(temp) {{
                if (temp > 7500) return {{ color: 0xa9c1ff, intensity: 2.5 }};
                if (temp > 6000) return {{ color: 0xfff4e8, intensity: 2 }};
                if (temp > 5200) return {{ color: 0xfff0a1, intensity: 1.8 }};
                if (temp > 3700) return {{ color: 0xffcc6f, intensity: 1.5 }};
                return {{ color: 0xff8c5a, intensity: 1.2 }};
            }}

            // View-specific scene setup
            {scene_setup}
            animate();

            // Handle window resize
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """

    return html_code

def create_solar_system_view(system_data, zoom_level='system'):
    """Create interactive solar system visualization with star and planet"""

    planet_radius = system_data.get('planet_radius')
    star_radius = system_data.get('star_radius')
    orbital_distance = system_data.get('orbital_distance')
    orbital_period = system_data.get('orbital_period')
    star_temp = system_data.get('star_temp')

    # Check for required data
    if pd.isna(planet_radius) or pd.isna(star_radius) or pd.isna(orbital_distance):
        return None

    # Classify planet and star
    if planet_radius < 1.5:
        planet_color = '#9F2B00'
        planet_type = "Terrestrial"
    elif planet_radius < 4:
        planet_color = '#AF4425'
        planet_type = "Super Earth"
    elif planet_radius < 10:
        planet_color = '#5B92E5'
        planet_type = "Neptune-like"
    else:
        planet_color = '#D2B48C'
        planet_type = "Gas Giant"

    star_type, star_color = classify_star_by_temp(star_temp)

    # Create mesh grids for spheres
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)

    # Create figure
    fig = go.Figure()

    if zoom_level == 'star':
        # Star comparison view: Show host star vs Sun
        star_display_radius = star_radius
        sun_display_radius = 1.0

        # Position: star on left, sun on right
        spacing = max(star_display_radius, sun_display_radius) * 2.5

        # Host star sphere (left)
        x_star = star_display_radius * np.outer(np.cos(u), np.sin(v)) - spacing/2
        y_star = star_display_radius * np.outer(np.sin(u), np.sin(v))
        z_star = star_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Sun sphere (right)
        x_sun = sun_display_radius * np.outer(np.cos(u), np.sin(v)) + spacing/2
        y_sun = sun_display_radius * np.outer(np.sin(u), np.sin(v))
        z_sun = sun_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_star, y=y_star, z=z_star,
            colorscale=[[0, star_color], [1, star_color]],
            showscale=False,
            name=f'{star_type} Star',
            hovertemplate=f'<b>{star_type} Host Star</b><br>Radius: {star_radius:.2f}x Sun<br>Temp: {star_temp:.0f} K<extra></extra>',
            lighting=dict(ambient=0.9, diffuse=0.9, fresnel=0.1)
        ))

        fig.add_trace(go.Surface(
            x=x_sun, y=y_sun, z=z_sun,
            colorscale=[[0, '#FDB813'], [1, '#FDB813']],
            showscale=False,
            name='Sun',
            hovertemplate='<b>Sun (for comparison)</b><br>Radius: 1.0x Sun<br>Temp: 5778 K<extra></extra>',
            lighting=dict(ambient=0.9, diffuse=0.9, fresnel=0.1)
        ))

        camera_distance = spacing * 1.2
        title = f'{star_type} Host Star vs Sun'

    elif zoom_level == 'planet':
        # Planet comparison view: Show exoplanet vs Earth
        planet_display_radius = planet_radius
        earth_display_radius = 1.0

        # Position: planet on left, Earth on right
        spacing = max(planet_display_radius, earth_display_radius) * 2.5

        # Exoplanet sphere (left)
        x_planet = planet_display_radius * np.outer(np.cos(u), np.sin(v)) - spacing/2
        y_planet = planet_display_radius * np.outer(np.sin(u), np.sin(v))
        z_planet = planet_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Earth sphere (right)
        x_earth = earth_display_radius * np.outer(np.cos(u), np.sin(v)) + spacing/2
        y_earth = earth_display_radius * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_planet, y=y_planet, z=z_planet,
            colorscale=[[0, planet_color], [1, planet_color]],
            showscale=False,
            name=f'{planet_type} Planet',
            hovertemplate=f'<b>{planet_type} Exoplanet</b><br>Radius: {planet_radius:.2f}x Earth<br>Period: {orbital_period:.1f} days<extra></extra>'
        ))

        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale=[[0, '#4F7CAC'], [1, '#4F7CAC']],
            showscale=False,
            name='Earth',
            hovertemplate='<b>Earth (for comparison)</b><br>Radius: 1.0x Earth<extra></extra>'
        ))

        camera_distance = spacing * 1.2
        title = f'{planet_type} Exoplanet vs Earth'

    else:  # system view
        # Full solar system view - use semi-fixed display distance
        star_display_radius = star_radius  # Use actual star radius for variation

        # Calculate actual planet radius ratio and scale for visibility
        # Convert planet radius (Earth radii) to same units as star (solar radii)
        planet_radius_in_solar = planet_radius * 0.00917  # Earth radii to solar radii
        planet_display_radius = planet_radius_in_solar

        # Enhance planet size for visibility (multiply by 10 so we can see it)
        # but keep it proportional to actual size
        planet_display_radius = planet_display_radius * 10.0

        # Cap planet at 50% of star size to keep it reasonable
        planet_display_radius = min(planet_display_radius, star_display_radius * 0.5)

        # Minimum size for visibility
        planet_display_radius = max(planet_display_radius, 0.05)

        # Display distance scales with square root of star size
        # This makes larger stars appear bigger without making orbit too large
        distance_display = np.sqrt(star_display_radius) * 2.5

        # Star sphere (at origin)
        x_star = star_display_radius * np.outer(np.cos(u), np.sin(v))
        y_star = star_display_radius * np.outer(np.sin(u), np.sin(v))
        z_star = star_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Planet sphere (at fixed display distance)
        x_planet = planet_display_radius * np.outer(np.cos(u), np.sin(v)) + distance_display
        y_planet = planet_display_radius * np.outer(np.sin(u), np.sin(v))
        z_planet = planet_display_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Create orbital path (circle at fixed distance)
        theta = np.linspace(0, 2 * np.pi, 100)
        orbit_x = distance_display * np.cos(theta)
        orbit_y = distance_display * np.sin(theta)
        orbit_z = np.zeros_like(theta)

        fig.add_trace(go.Surface(
            x=x_star, y=y_star, z=z_star,
            colorscale=[[0, star_color], [1, star_color]],
            showscale=False,
            name=f'{star_type} Star',
            hovertemplate=f'<b>{star_type} Star</b><br>Radius: {star_radius:.2f}x Sun<br>Temp: {star_temp:.0f} K<extra></extra>',
            lighting=dict(ambient=0.9, diffuse=0.9, fresnel=0.1)
        ))

        fig.add_trace(go.Surface(
            x=x_planet, y=y_planet, z=z_planet,
            colorscale=[[0, planet_color], [1, planet_color]],
            showscale=False,
            name=f'{planet_type} Planet',
            hovertemplate=f'<b>{planet_type}</b><br>Radius: {planet_radius:.2f}x Earth<br>Period: {orbital_period:.1f} days<extra></extra>'
        ))

        fig.add_trace(go.Scatter3d(
            x=orbit_x, y=orbit_y, z=orbit_z,
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add text label pointing to planet location
        fig.add_trace(go.Scatter3d(
            x=[distance_display],
            y=[0],
            z=[planet_display_radius + 0.5],
            mode='text',
            text=[f'{planet_type} Planet'],
            textposition='top center',
            textfont=dict(size=14, color='#60A5FA'),
            showlegend=False,
            hoverinfo='skip'
        ))

        camera_distance = max(distance_display * 1.5, star_display_radius * 3)
        title = f'{star_type} Star with {planet_type} Planet'

    # Common layout settings
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False, range=[-camera_distance, camera_distance]),
            yaxis=dict(visible=False, range=[-camera_distance, camera_distance]),
            zaxis=dict(visible=False, range=[-camera_distance, camera_distance]),
            bgcolor='#000000',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        paper_bgcolor='#1F2937',
        plot_bgcolor='#000000',
        font=dict(color='#F3F4F6'),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    return {
        'fig': fig,
        'planet_type': planet_type,
        'star_type': star_type,
        'planet_radius': planet_radius,
        'star_radius': star_radius,
        'orbital_distance': orbital_distance,
        'orbital_period': orbital_period,
        'star_temp': star_temp
    }

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

    # Load system data from raw dataset
    system_data_all = load_raw_planet_data()

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Sample Explorer", "🔮 Candidate Predictor", "📂 Import & Predict", "📊 Data Explorer", "📈 Model Performance", "📚 Documentation"])

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

            # Section 2: 3D Solar System Visualization
            elif view_section == "3D Visualization":
                st.subheader("🌌 Solar System Visualization")

                # Initialize zoom level in session state
                if 'zoom_level' not in st.session_state:
                    st.session_state.zoom_level = 'system'

                # View controls (before visualization so they can update state)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🌟 View Star", key="zoom_star"):
                        st.session_state.zoom_level = 'star'
                        st.rerun()
                with col2:
                    if st.button("🌍 View Planet", key="zoom_planet"):
                        st.session_state.zoom_level = 'planet'
                        st.rerun()
                with col3:
                    if st.button("🔭 Full System View", key="zoom_system"):
                        st.session_state.zoom_level = 'system'
                        st.rerun()

                system_data = get_system_data(sample_idx, system_data_all)

                if system_data and not pd.isna(system_data.get('planet_radius')):
                    # Generate and display Three.js visualization
                    html_viz = create_threejs_visualization(system_data, st.session_state.zoom_level)
                    components.html(html_viz, height=600, scrolling=False)

                    # Also get data for metrics display
                    viz_result = create_solar_system_view(system_data, st.session_state.zoom_level)
                else:
                    viz_result = None

                if viz_result:

                    # System information
                    st.markdown("### System Information")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Star Type", viz_result['star_type'])
                        st.caption(f"Temp: {viz_result['star_temp']:.0f} K" if not pd.isna(viz_result['star_temp']) else "Temp: N/A")

                    with col2:
                        st.metric("Star Radius", f"{viz_result['star_radius']:.2f}x Sun")
                        st.caption("Compared to our Sun")

                    with col3:
                        st.metric("Planet Type", viz_result['planet_type'])
                        st.metric("Planet Radius", f"{viz_result['planet_radius']:.2f}x Earth")

                    with col4:
                        st.metric("Orbital Distance", f"{viz_result['orbital_distance']:.3f} AU")
                        st.metric("Orbital Period", f"{viz_result['orbital_period']:.1f} days")

                    st.caption("**AU** = Astronomical Unit (Earth-Sun distance, ~93 million miles)")

                else:
                    st.info("System data not available for this KOI.")
                    st.markdown("""
                    **Note:** Complete system data (star radius, planet radius, orbital distance) is only
                    available for confirmed planets and candidates. False positives typically don't have
                    reliable orbital parameters.
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
        st.header("🔮 Candidate KOI Predictor")
        st.markdown("""
        This tool analyzes **unconfirmed candidate** KOIs from the Kepler dataset and predicts
        whether they are likely to be real exoplanets or false positives based on the trained model.
        """)

        # Load candidate data
        X_candidates, metadata, original_indices = load_candidate_kois(model_type)

        if X_candidates is not None and metadata is not None:
            st.success(f"✅ Loaded {len(X_candidates)} candidate KOIs for analysis")

            # Overview metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Candidates", len(X_candidates))
            with col2:
                avg_period = metadata['koi_period'].mean()
                st.metric("Avg Orbital Period", f"{avg_period:.1f} days")
            with col3:
                avg_depth = metadata['koi_depth'].mean()
                st.metric("Avg Transit Depth", f"{avg_depth:.0f} ppm")

            # Classify all button
            if st.button("🔮 Classify All Candidates", type="primary", key="classify_all_candidates"):
                with st.spinner("Analyzing all candidates..."):
                    # Make predictions
                    predictions = model.predict(X_candidates)
                    probabilities = model.predict_proba(X_candidates)[:, 1]

                    # Add predictions to metadata
                    results_df = metadata.copy()
                    results_df['Prediction'] = ['Planet' if p == 1 else 'Non-Planet' for p in predictions]
                    results_df['Planet_Probability'] = probabilities
                    results_df['Confidence'] = np.maximum(probabilities, 1 - probabilities)

                    # Sort by confidence (most confident predictions first)
                    results_df = results_df.sort_values('Confidence', ascending=False)

                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Classification Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        planet_count = sum(predictions)
                        st.metric("Predicted Planets", planet_count)
                    with col2:
                        non_planet_count = len(predictions) - planet_count
                        st.metric("Predicted Non-Planets", non_planet_count)
                    with col3:
                        high_conf_planets = sum((predictions == 1) & (probabilities > 0.9))
                        st.metric("High Confidence Planets", high_conf_planets)
                        st.caption("(>90% probability)")
                    with col4:
                        avg_conf = results_df['Confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")

                    # Confidence distribution
                    st.markdown("### Confidence Distribution")
                    fig_conf = go.Figure()
                    fig_conf.add_trace(go.Histogram(
                        x=probabilities,
                        nbinsx=30,
                        name="Planet Probability",
                        marker_color='#60a5fa'
                    ))
                    fig_conf.update_layout(
                        xaxis_title="Planet Probability",
                        yaxis_title="Number of Candidates",
                        height=300,
                        paper_bgcolor='#1F2937',
                        plot_bgcolor='#374151',
                        font=dict(color='#F3F4F6')
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

                    # Results table
                    st.markdown("### 🎯 Top Predictions (Sorted by Confidence)")
                    st.markdown("**High-confidence predictions** are shown first. These are the most reliable classifications.")

                    # Format the display
                    display_df = results_df[['kepoi_name', 'Prediction', 'Planet_Probability', 'Confidence', 'koi_period', 'koi_depth', 'koi_prad', 'koi_steff']].copy()
                    display_df.columns = ['KOI Name', 'Prediction', 'Planet Prob.', 'Confidence', 'Period (days)', 'Depth (ppm)', 'Radius (Earth)', 'Star Temp (K)']

                    # Color code by prediction
                    def highlight_prediction(row):
                        if row['Prediction'] == 'Planet':
                            return ['background-color: #1e3a5f'] * len(row)
                        else:
                            return ['background-color: #3a1e1e'] * len(row)

                    st.dataframe(
                        display_df.head(50).style.format({
                            'Planet Prob.': '{:.1%}',
                            'Confidence': '{:.1%}',
                            'Period (days)': '{:.2f}',
                            'Depth (ppm)': '{:.0f}',
                            'Radius (Earth)': '{:.2f}',
                            'Star Temp (K)': '{:.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results as CSV",
                        data=csv,
                        file_name=f"candidate_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    # Interesting findings
                    st.markdown("---")
                    st.markdown("### 🌟 Noteworthy Findings")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🌍 Top 5 Most Likely Planets:**")
                        top_planets = results_df[results_df['Prediction'] == 'Planet'].head(5)
                        for idx, row in top_planets.iterrows():
                            st.markdown(f"- **{row['kepoi_name']}**: {row['Planet_Probability']:.1%} confidence")

                    with col2:
                        st.markdown("**❌ Top 5 Most Likely False Positives:**")
                        top_non_planets = results_df[results_df['Prediction'] == 'Non-Planet'].head(5)
                        for idx, row in top_non_planets.iterrows():
                            st.markdown(f"- **{row['kepoi_name']}**: {(1-row['Planet_Probability']):.1%} confidence")

        else:
            st.warning("⚠️ Could not load candidate KOIs. Please ensure the dataset is available.")

    with tab3:
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

    with tab4:
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

    with tab5:
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

    with tab6:
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
