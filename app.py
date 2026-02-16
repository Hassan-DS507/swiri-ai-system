import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components


# ----------------------------------------------------------------------
# SECTION 1: PAGE CONFIGURATION AND INITIAL SETUP
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="SWIRI AI Safety System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# SECTION 2: SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------

if 'heart_rate_raw' not in st.session_state:
    st.session_state.heart_rate_raw = []
if 'accelerometer_raw' not in st.session_state:
    st.session_state.accelerometer_raw = []
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'features' not in st.session_state:
    st.session_state.features = {}
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = None
if 'location' not in st.session_state:
    st.session_state.location = "School Playground"
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'parent_confirmation' not in st.session_state:
    st.session_state.parent_confirmation = None
if 'event_logs' not in st.session_state:
    st.session_state.event_logs = []
if 'scenario' not in st.session_state:
    st.session_state.scenario = "None"
if 'alert_triggered' not in st.session_state:
    st.session_state.alert_triggered = False

# ----------------------------------------------------------------------
# SECTION 3: CUSTOM CSS STYLING
# ----------------------------------------------------------------------

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E3A8A;
        font-weight: 600;
    }
    
    /* Status banners */
    .safe-banner {
        background-color: #10B981;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background-color: #F59E0B;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .danger-banner {
        background-color: #EF4444;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        animation: pulse 1.5s infinite;
    }
    
    /* Cards */
    .info-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E3A8A;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-bottom: 3px solid #1E3A8A;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes waveLeft {
        0% { transform: rotate(-20deg); }
        50% { transform: rotate(-30deg); }
        100% { transform: rotate(-20deg); }
    }
    
    @keyframes waveRight {
        0% { transform: rotate(20deg); }
        50% { transform: rotate(30deg); }
        100% { transform: rotate(20deg); }
    }
    
    @keyframes walkLeft {
        0% { transform: rotate(2deg) translateY(0); }
        50% { transform: rotate(4deg) translateY(-3px); }
        100% { transform: rotate(2deg) translateY(0); }
    }
    
    @keyframes walkRight {
        0% { transform: rotate(-2deg) translateY(0); }
        50% { transform: rotate(-4deg) translateY(-3px); }
        100% { transform: rotate(-2deg) translateY(0); }
    }
    
    @keyframes nodHead {
        0% { transform: rotate(0deg); }
        25% { transform: rotate(5deg); }
        75% { transform: rotate(-5deg); }
        100% { transform: rotate(0deg); }
    }
    
    @keyframes watchGlow {
        0% { box-shadow: 0 0 5px #1E3A8A; }
        50% { box-shadow: 0 0 20px #1E3A8A; }
        100% { box-shadow: 0 0 5px #1E3A8A; }
    }
    
    /* Animation classes */
    .child-container {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Child illustration container */
    .child-drawing {
        position: relative;
        width: 100%;
        height: 450px;
        background-color: #f8f9fa;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        border: 2px solid #1E3A8A;
    }
    @keyframes blink {
    0%, 48%, 52%, 100% { transform: scaleY(1); }
    50% { transform: scaleY(0.1); }
}

@keyframes smile {
    0%, 100% { border-bottom-width: 4px; }
    50% { border-bottom-width: 6px; }
}

@keyframes shadow {
    0%, 100% { transform: scale(1); opacity: 0.2; }
    50% { transform: scale(1.1); opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SECTION 4: NAVIGATION MENU
# ----------------------------------------------------------------------

def create_nav_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Smartwatch", "AI Engine", "Notifications", "Confirmation", "Logs"],
        icons=["house", "watch", "cpu", "bell", "check-circle", "journal-text"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1E3A8A", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1E3A8A"},
        }
    )
    return selected

page = create_nav_menu()

# ----------------------------------------------------------------------
# SECTION 5: MODEL LOADING
# ----------------------------------------------------------------------

@st.cache_resource
def load_model():
    try:
        model = joblib.load('swiri_rf_model.pkl')
        return model
    except:
        st.error("Model file not found. Please ensure 'swiri_rf_model.pkl' is in the current directory.")
        return None

model = load_model()

# ----------------------------------------------------------------------
# SECTION 6: HELPER FUNCTIONS
# ----------------------------------------------------------------------

def generate_sensor_data(scenario):
    np.random.seed(int(time.time()))
    
    if scenario == "NORMAL":
        hr_base = 85
        hr_range = (80, 95)
        hr_trend = np.random.normal(0, 0.5, 50)
        acc_base = 0.5
        acc_variance = 0.1
    elif scenario == "PLAYING":
        hr_base = 110
        hr_range = (100, 120)
        hr_trend = np.random.normal(0, 1, 50) + np.linspace(0, 5, 50)
        acc_base = 2.0
        acc_variance = 0.5
    else:
        hr_base = 145
        hr_range = (130, 160)
        hr_trend = np.random.normal(0, 2, 50) + np.linspace(0, 15, 50)
        acc_base = 4.0
        acc_variance = 1.5
    
    heart_rate = hr_base + hr_trend + np.random.normal(0, 2, 50)
    heart_rate = np.clip(heart_rate, hr_range[0], hr_range[1])
    
    accelerometer = acc_base + np.random.normal(0, acc_variance, 50)
    accelerometer = np.clip(accelerometer, 0, 10)
    
    return heart_rate.tolist(), accelerometer.tolist()

def compute_features(hr_data, acc_data):
    features = {
        'hr_mean': np.mean(hr_data),
        'hr_gradient': hr_data[-1] - hr_data[0],
        'acc_mean': np.mean(acc_data),
        'acc_variance': np.var(acc_data)
    }
    return features

def get_prediction_label(pred):
    labels = {0: "NORMAL", 1: "PLAYING", 2: "DANGER"}
    return labels.get(pred, "UNKNOWN")

def get_watch_color(pred):
    colors = {0: "#10B981", 1: "#F59E0B", 2: "#EF4444"}
    return colors.get(pred, "#1E3A8A")

def log_event(event_type, details):
    st.session_state.event_logs.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'type': event_type,
        'details': details
    })

# ----------------------------------------------------------------------
# SECTION 7: HOME PAGE
# ----------------------------------------------------------------------

if page == "Home":
    st.title("SWIRI AI Safety System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>System Overview</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                SWIRI is an intelligent child safety monitoring system that uses wearable sensor data 
                and machine learning to detect potential dangerous situations in real-time. The system 
                continuously monitors heart rate and movement patterns to classify child activities 
                into three categories: Normal, Playing, or Danger.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### System Architecture")
        st.code("""
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│  Wearable   │───▶│   Sensor    │───▶│    Feature      │
│   Device    │    │    Data     │    │  Engineering    │
└─────────────┘    └─────────────┘    └─────────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Parent    │◀───│    Alert    │◀───│   AI Model      │
│ Confirmation│    │   System    │    │  Classification │
└─────────────┘    └─────────────┘    └─────────────────┘
      │                   ▲
      ▼                   │
┌─────────────┐    ┌─────────────┐
│  Feedback   │────│   School    │
│    Loop     │    │ Notification│
└─────────────┘    └─────────────┘
        """)
    
    with col2:
        st.markdown("### Technical Specs")
        st.markdown("""
        <div class="metric-box">
            <h4>Sensor Sampling Rate</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">10 Hz</p>
        </div>
        <br>
        <div class="metric-box">
            <h4>Window Size</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">5 seconds</p>
        </div>
        <br>
        <div class="metric-box">
            <h4>Samples per Window</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">50</p>
        </div>
        <br>
        <div class="metric-box">
            <h4>Model Type</h4>
            <p style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">Random Forest</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SECTION 8: SMARTWATCH SIMULATION PAGE
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# SECTION 8: SMARTWATCH SIMULATION PAGE
# ----------------------------------------------------------------------

elif page == "Smartwatch":
    st.title("Smartwatch Simulation")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Child Monitor View")
        
        # Get watch color based on prediction
        watch_color = get_watch_color(st.session_state.prediction) if st.session_state.prediction is not None else "#10B981"
        
        # Determine which scene to show
        scene = st.session_state.scenario if st.session_state.scenario else "NORMAL"
        
        # Create realistic child illustration HTML with multiple scenes
        child_html = f"""
        <div style="position: relative; width: 100%; background: linear-gradient(135deg, #87CEEB 0%, #98D8E8 100%); border-radius: 20px; padding: 30px 20px; border: 4px solid #1E3A8A; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            
            <!-- Background scene changes based on scenario -->
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; border-radius: 16px;">
                <!-- Sky with sun or clouds based on scenario -->
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 70%; 
                            background: {'linear-gradient(to bottom, #87CEEB, #B0E0E6)' if scene == 'NORMAL' else 'linear-gradient(to bottom, #FFD700, #87CEEB)' if scene == 'PLAYING' else 'linear-gradient(to bottom, #2C3E50, #34495E)'};">
                </div>
                
                <!-- Sun or moon based on scenario -->
                <div style="position: absolute; top: 20px; right: 20px; width: 60px; height: 60px; 
                            background: {'#FFD700' if scene != 'DANGER' else '#95A5A6'}; 
                            border-radius: 50%; 
                            box-shadow: {'0 0 30px #FFD700' if scene != 'DANGER' else '0 0 30px #95A5A6'};
                            animation: {'sunRotate 10s linear infinite' if scene == 'PLAYING' else 'none'};">
                </div>
                
                <!-- Clouds (only in normal and playing) -->
                {'<div style="position: absolute; top: 50px; left: 30px; width: 80px; height: 30px; background: white; border-radius: 30px; opacity: 0.8; animation: cloudMove 15s linear infinite;"></div>' if scene != 'DANGER' else ''}
                {'<div style="position: absolute; top: 80px; left: 150px; width: 100px; height: 35px; background: white; border-radius: 35px; opacity: 0.6; animation: cloudMove 20s linear infinite;"></div>' if scene != 'DANGER' else ''}
                
                <!-- Ground -->
                <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 30%; 
                            background: {'linear-gradient(to top, #2ecc71, #27ae60)' if scene != 'DANGER' else 'linear-gradient(to top, #7f8c8d, #95a5a6)'};
                            border-top: 3px solid #1E3A8A;">
                    
                    <!-- Grass details -->
                    {'<div style="position: absolute; top: -10px; left: 10px; width: 5px; height: 15px; background: #27ae60; transform: rotate(10deg);"></div>' * 20 if scene != 'DANGER' else ''}
                </div>
            </div>
            
            <!-- Main character container -->
            <div style="position: relative; width: 400px; height: 500px; margin: 0 auto;" class="child-container">
                
                <!-- SCENE 1: NORMAL (Standing calmly, green watch) -->
                <div style="position: relative; display: {'block' if scene == 'NORMAL' else 'none'};">
                    <!-- Child standing normally -->
                    <div class="child-standing">
                        <!-- Head -->
                        <div style="position: absolute; top: 20px; left: 150px; width: 90px; height: 90px; 
                                    background: radial-gradient(circle at 30% 30%, #FFE4B5, #DEB887); 
                                    border-radius: 50%; border: 3px solid #1E3A8A;
                                    box-shadow: 0 8px 15px rgba(0,0,0,0.2);">
                            <!-- Happy face -->
                            <div style="position: absolute; top: 50px; left: 55px; width: 30px; height: 15px; 
                                        border-bottom: 4px solid #1E3A8A; border-radius: 0 0 30px 30px;"></div>
                            <div style="position: absolute; top: 30px; left: 45px; width: 8px; height: 8px; 
                                        background: #1E3A8A; border-radius: 50%;"></div>
                            <div style="position: absolute; top: 30px; left: 75px; width: 8px; height: 8px; 
                                        background: #1E3A8A; border-radius: 50%;"></div>
                        </div>
                        
                        <!-- Body -->
                        <div style="position: absolute; top: 105px; left: 165px; width: 60px; height: 100px; 
                                    background: #3498db; border-radius: 15px; border: 3px solid #1E3A8A;">
                        </div>
                        
                        <!-- Arms down -->
                        <div style="position: absolute; top: 120px; left: 125px; width: 45px; height: 20px; 
                                    background: #3498db; border-radius: 20px; transform: rotate(-10deg); border: 3px solid #1E3A8A;"></div>
                        <div style="position: absolute; top: 120px; left: 215px; width: 45px; height: 20px; 
                                    background: #3498db; border-radius: 20px; transform: rotate(10deg); border: 3px solid #1E3A8A;"></div>
                        
                        <!-- Green Watch -->
                        <div style="position: absolute; top: 115px; left: 245px; width: 35px; height: 35px; 
                                    background: #10B981; border-radius: 10px; border: 3px solid #1E3A8A;
                                    box-shadow: 0 0 20px #10B981; animation: watchPulse 2s infinite;">
                            <div style="position: absolute; top: 8px; left: 8px; width: 15px; height: 15px; 
                                        background: white; border-radius: 5px;"></div>
                        </div>
                        
                        <!-- Legs -->
                        <div style="position: absolute; top: 200px; left: 175px; width: 20px; height: 70px; 
                                    background: #2c3e50; border-radius: 10px; border: 3px solid #1E3A8A;"></div>
                        <div style="position: absolute; top: 200px; left: 195px; width: 20px; height: 70px; 
                                    background: #2c3e50; border-radius: 10px; border: 3px solid #1E3A8A;"></div>
                    </div>
                    
                    <!-- Status text -->
                    <div style="position: absolute; top: 350px; left: 150px; background: #10B981; 
                                padding: 10px 20px; border-radius: 20px; color: white; font-weight: bold;
                                border: 2px solid #1E3A8A;">
                        NORMAL - Safe & Calm 🏠
                    </div>
                </div>
                
                <!-- SCENE 2: PLAYING (Running with football, yellow watch) -->
                <div style="position: relative; display: {'block' if scene == 'PLAYING' else 'none'};">
                    <!-- Child running with football -->
                    <div class="child-playing">
                        <!-- Head (excited expression) -->
                        <div style="position: absolute; top: 20px; left: 150px; width: 90px; height: 90px; 
                                    background: radial-gradient(circle at 30% 30%, #FFE4B5, #DEB887); 
                                    border-radius: 50%; border: 3px solid #1E3A8A;
                                    animation: headBob 0.5s infinite;">
                            <!-- Excited face (open mouth) -->
                            <div style="position: absolute; top: 50px; left: 55px; width: 20px; height: 20px; 
                                        background: #FF6B6B; border-radius: 50%; border: 2px solid #1E3A8A;"></div>
                            <div style="position: absolute; top: 30px; left: 45px; width: 10px; height: 10px; 
                                        background: #1E3A8A; border-radius: 50%;"></div>
                            <div style="position: absolute; top: 30px; left: 75px; width: 10px; height: 10px; 
                                        background: #1E3A8A; border-radius: 50%;"></div>
                            <!-- Sweat -->
                            <div style="position: absolute; top: 20px; left: 80px; width: 5px; height: 10px; 
                                        background: #87CEEB; border-radius: 5px; transform: rotate(20deg);"></div>
                        </div>
                        
                        <!-- Body leaning forward -->
                        <div style="position: absolute; top: 105px; left: 165px; width: 60px; height: 90px; 
                                    background: #e74c3c; border-radius: 15px; border: 3px solid #1E3A8A;
                                    transform: rotate(5deg);">
                            <!-- Number on shirt -->
                            <div style="position: absolute; top: 30px; left: 20px; color: white; 
                                        font-weight: bold; font-size: 20px;">10</div>
                        </div>
                        
                        <!-- Arms running -->
                        <div style="position: absolute; top: 115px; left: 120px; width: 50px; height: 20px; 
                                    background: #e74c3c; border-radius: 20px; transform: rotate(-45deg); 
                                    border: 3px solid #1E3A8A; animation: armRunLeft 0.5s infinite;"></div>
                        <div style="position: absolute; top: 115px; left: 215px; width: 50px; height: 20px; 
                                    background: #e74c3c; border-radius: 20px; transform: rotate(45deg); 
                                    border: 3px solid #1E3A8A; animation: armRunRight 0.5s infinite;"></div>
                        
                        <!-- Yellow Watch -->
                        <div style="position: absolute; top: 110px; left: 250px; width: 35px; height: 35px; 
                                    background: #F59E0B; border-radius: 10px; border: 3px solid #1E3A8A;
                                    box-shadow: 0 0 20px #F59E0B; animation: watchGlow 0.5s infinite;">
                            <div style="position: absolute; top: 8px; left: 8px; color: white; 
                                        font-size: 12px; font-weight: bold;">⚽</div>
                        </div>
                        
                        <!-- Legs running -->
                        <div style="position: absolute; top: 190px; left: 175px; width: 20px; height: 80px; 
                                    background: #2c3e50; border-radius: 10px; border: 3px solid #1E3A8A;
                                    transform: rotate(15deg); animation: legRun 0.5s infinite;"></div>
                        <div style="position: absolute; top: 190px; left: 195px; width: 20px; height: 80px; 
                                    background: #2c3e50; border-radius: 10px; border: 3px solid #1E3A8A;
                                    transform: rotate(-15deg); animation: legRun 0.5s infinite reverse;"></div>
                        
                        <!-- Football -->
                        <div style="position: absolute; top: 200px; left: 230px; width: 30px; height: 30px; 
                                    background: #8B4513; border-radius: 50%; border: 3px solid #1E3A8A;
                                    animation: ballRoll 0.3s infinite;">
                            <div style="position: absolute; top: 5px; left: 5px; width: 20px; height: 20px; 
                                        border: 2px solid black; border-radius: 50%;"></div>
                        </div>
                        
                        <!-- Dust particles -->
                        <div style="position: absolute; top: 220px; left: 150px; width: 5px; height: 5px; 
                                    background: #95a5a6; border-radius: 50%; animation: dust 0.5s infinite;"></div>
                        <div style="position: absolute; top: 225px; left: 160px; width: 8px; height: 8px; 
                                    background: #95a5a6; border-radius: 50%; animation: dust 0.7s infinite;"></div>
                    </div>
                    
                    <!-- Status text -->
                    <div style="position: absolute; top: 350px; left: 150px; background: #F59E0B; 
                                padding: 10px 20px; border-radius: 20px; color: white; font-weight: bold;
                                border: 2px solid #1E3A8A;">
                        PLAYING - Running & Playing ⚽
                    </div>
                </div>
                
          
                <div style="position: relative; display: {'block' if scene == 'DANGER' else 'none'};">
                    <!-- Child being grabbed -->
                    <div class="child-danger">
                        <!-- Child with scared expression -->
                        <div style="position: absolute; top: 20px; left: 120px; width: 90px; height: 90px; 
                                    background: radial-gradient(circle at 30% 30%, #FFE4B5, #DEB887); 
                                    border-radius: 50%; border: 3px solid #1E3A8A;
                                    animation: shake 0.1s infinite;">
                            <!-- Scared face -->
                            <div style="position: absolute; top: 35px; left: 40px; width: 15px; height: 15px; 
                                        background: white; border-radius: 50%; border: 3px solid #1E3A8A;"></div>
                            <div style="position: absolute; top: 35px; left: 70px; width: 15px; height: 15px; 
                                        background: white; border-radius: 50%; border: 3px solid #1E3A8A;"></div>
                            <div style="position: absolute; top: 40px; left: 45px; width: 5px; height: 5px; 
                                        background: #1E3A8A; border-radius: 50%;"></div>
                            <div style="position: absolute; top: 40px; left: 75px; width: 5px; height: 5px; 
                                        background: #1E3A8A; border-radius: 50%;"></div>
                            <!-- Open mouth screaming -->
                            <div style="position: absolute; top: 60px; left: 55px; width: 20px; height: 20px; 
                                        background: #EF4444; border-radius: 50%; border: 2px solid #1E3A8A;"></div>
                            <!-- Tears -->
                            <div style="position: absolute; top: 45px; left: 35px; width: 5px; height: 10px; 
                                        background: #87CEEB; border-radius: 5px; animation: tearDrop 0.5s infinite;"></div>
                        </div>
                        
                        <!-- Child's body (leaning back in fear) -->
                        <div style="position: absolute; top: 105px; left: 135px; width: 60px; height: 90px; 
                                    background: #3498db; border-radius: 15px; border: 3px solid #1E3A8A;
                                    transform: rotate(-10deg);">
                        </div>
                        
                        <!-- Child's arms (trying to resist) -->
                        <div style="position: absolute; top: 115px; left: 95px; width: 45px; height: 20px; 
                                    background: #3498db; border-radius: 20px; transform: rotate(30deg); 
                                    border: 3px solid #1E3A8A; animation: struggle 0.2s infinite;"></div>
                        <div style="position: absolute; top: 115px; left: 185px; width: 45px; height: 20px; 
                                    background: #3498db; border-radius: 20px; transform: rotate(-20deg); 
                                    border: 3px solid #1E3A8A; animation: struggle 0.2s infinite reverse;"></div>
                        
                        <!-- Red Watch (flashing) -->
                        <div style="position: absolute; top: 110px; left: 215px; width: 40px; height: 40px; 
                                    background: #EF4444; border-radius: 10px; border: 3px solid #1E3A8A;
                                    box-shadow: 0 0 30px #EF4444; animation: dangerFlash 0.1s infinite;">
                            <div style="position: absolute; top: 5px; left: 5px; color: white; 
                                        font-size: 20px; font-weight: bold; animation: pulse 0.1s infinite;">SOS</div>
                        </div>
                        
               
                        <!-- Kidnapper's body -->
                        <div style="position: absolute; top: 90px; left: 190px; width: 80px; height: 120px; 
                                    background: #2C3E50; border-radius: 20px; border: 4px solid #1E3A8A;
                                    transform: rotate(5deg);">
                            <!-- "KIDNAPPER" text on shirt -->
                            <div style="position: absolute; top: 40px; left: 10px; color: #EF4444; 
                                        font-weight: bold; font-size: 12px; transform: rotate(-5deg);">???</div>
                        </div>
                        
                        <!-- Kidnapper's head (scary) -->
                        <div style="position: absolute; top: 20px; left: 210px; width: 80px; height: 80px; 
                                    background: #4A5568; border-radius: 50%; border: 4px solid #1E3A8A;
                                    filter: drop-shadow(0 0 10px #EF4444);">
                            <!-- Scary mask -->
                            <div style="position: absolute; top: 20px; left: 15px; width: 20px; height: 10px; 
                                        background: #EF4444; border-radius: 10px;"></div>
                            <div style="position: absolute; top: 20px; left: 45px; width: 20px; height: 10px; 
                                        background: #EF4444; border-radius: 10px;"></div>
                            <div style="position: absolute; top: 40px; left: 25px; width: 30px; height: 20px; 
                                        background: #1E3A8A; border-radius: 10px;"></div>
                            <!-- Menacing eyes -->
                            <div style="position: absolute; top: 25px; left: 25px; width: 5px; height: 5px; 
                                        background: #EF4444; border-radius: 50%; animation: glow 0.1s infinite;"></div>
                            <div style="position: absolute; top: 25px; left: 50px; width: 5px; height: 5px; 
                                        background: #EF4444; border-radius: 50%; animation: glow 0.1s infinite;"></div>
                        </div>
                        
                        <!-- Kidnapper's arm grabbing child -->
                        <div style="position: absolute; top: 110px; left: 165px; width: 60px; height: 25px; 
                                    background: #4A5568; border-radius: 30px; transform: rotate(-20deg); 
                                    border: 4px solid #1E3A8A; box-shadow: 0 0 15px #EF4444;">
                            <!-- Hand grabbing -->
                            <div style="position: absolute; top: -5px; left: -10px; width: 25px; height: 30px; 
                                        background: #4A5568; border-radius: 50% 50% 30% 30%; 
                                        border: 4px solid #1E3A8A;"></div>
                        </div>
                        
                        <!-- Kidnapper's other arm -->
                        <div style="position: absolute; top: 120px; left: 260px; width: 50px; height: 25px; 
                                    background: #4A5568; border-radius: 30px; transform: rotate(30deg); 
                                    border: 4px solid #1E3A8A;"></div>
                        
                        <!-- Kidnapper's legs (running stance) -->
                        <div style="position: absolute; top: 205px; left: 210px; width: 25px; height: 80px; 
                                    background: #2C3E50; border-radius: 10px; border: 4px solid #1E3A8A;
                                    transform: rotate(10deg); animation: kidnapperRun 0.3s infinite;"></div>
                        <div style="position: absolute; top: 205px; left: 235px; width: 25px; height: 80px; 
                                    background: #2C3E50; border-radius: 10px; border: 4px solid #1E3A8A;
                                    transform: rotate(-10deg); animation: kidnapperRun 0.3s infinite reverse;"></div>
                        
                        <!-- Danger symbols -->
                        <div style="position: absolute; top: 10px; left: 50px; color: #EF4444; 
                                    font-size: 30px; font-weight: bold; animation: dangerSign 0.2s infinite;">⚠️</div>
                        <div style="position: absolute; top: 300px; left: 100px; color: #EF4444; 
                                    font-size: 20px; animation: dangerSign 0.3s infinite;">🚨</div>
                    </div>
                    
                    <!-- Status text with alarm -->
                    <div style="position: absolute; top: 350px; left: 120px; background: #EF4444; 
                                padding: 15px 25px; border-radius: 30px; color: white; font-weight: bold;
                                border: 3px solid #1E3A8A; animation: alarmPulse 0.1s infinite;
                                font-size: 1.2rem;">
                        🚨 DANGER - KIDNAPPING ATTEMPT! 🚨
                    </div>
                </div>
            </div>
        </div>

        <style>
            /* Scene 2 animations (playing) */
            @keyframes armRunLeft {{
                0%, 100% {{ transform: rotate(-45deg); }}
                50% {{ transform: rotate(-60deg); }}
            }}
            
            @keyframes armRunRight {{
                0%, 100% {{ transform: rotate(45deg); }}
                50% {{ transform: rotate(60deg); }}
            }}
            
            @keyframes legRun {{
                0%, 100% {{ transform: rotate(15deg) translateY(0); }}
                50% {{ transform: rotate(25deg) translateY(-5px); }}
            }}
            
            @keyframes headBob {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-5px); }}
            }}
            
            @keyframes ballRoll {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            @keyframes dust {{
                0% {{ opacity: 1; transform: scale(1); }}
                100% {{ opacity: 0; transform: scale(2) translateX(20px); }}
            }}
            
            /* Scene 3 animations (danger) */
            @keyframes shake {{
                0%, 100% {{ transform: translateX(0); }}
                25% {{ transform: translateX(-5px); }}
                75% {{ transform: translateX(5px); }}
            }}
            
            @keyframes struggle {{
                0%, 100% {{ transform: rotate(30deg); }}
                50% {{ transform: rotate(45deg); }}
            }}
            
            @keyframes dangerFlash {{
                0%, 100% {{ background: #EF4444; box-shadow: 0 0 30px #EF4444; }}
                50% {{ background: #FF6B6B; box-shadow: 0 0 50px #FF0000; }}
            }}
            
            @keyframes alarmPulse {{
                0%, 100% {{ transform: scale(1); background: #EF4444; }}
                50% {{ transform: scale(1.1); background: #FF0000; }}
            }}
            
            @keyframes tearDrop {{
                0% {{ transform: translateY(0); opacity: 1; }}
                100% {{ transform: translateY(20px); opacity: 0; }}
            }}
            
            @keyframes glow {{
                0%, 100% {{ box-shadow: 0 0 5px #EF4444; }}
                50% {{ box-shadow: 0 0 20px #EF4444; }}
            }}
            
            @keyframes kidnapperRun {{
                0%, 100% {{ transform: rotate(10deg) translateY(0); }}
                50% {{ transform: rotate(15deg) translateY(-5px); }}
            }}
            
            @keyframes dangerSign {{
                0%, 100% {{ opacity: 1; transform: scale(1); }}
                50% {{ opacity: 0.5; transform: scale(1.5); }}
            }}
            
            @keyframes sunRotate {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            @keyframes cloudMove {{
                0% {{ transform: translateX(-100px); }}
                100% {{ transform: translateX(400px); }}
            }}
            
            .child-container {{
                animation: float 3s ease-in-out infinite;
            }}
        </style>
        """
        
        st.components.v1.html(child_html, height=650)
        
        # Show status with appropriate styling
        if st.session_state.prediction is not None or st.session_state.scenario:
            status = st.session_state.scenario if st.session_state.scenario else "NORMAL"
            if st.session_state.prediction is not None:
                status = get_prediction_label(st.session_state.prediction)
            
            color = {"NORMAL": "#10B981", "PLAYING": "#F59E0B", "DANGER": "#EF4444"}.get(status, "#1E3A8A")
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background-color: {color}; color: white; 
                        border-radius: 10px; font-weight: bold; font-size: 1.2rem; margin-top: 10px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                        animation: {'pulse 1s infinite' if status == 'DANGER' else 'none'};">
                Current Status: {status} 
                <span style="font-size: 1.5rem; margin-left: 10px;">
                    {'🟢' if status == 'NORMAL' else '🟡' if status == 'PLAYING' else '🔴🚨'}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Scenario Controls")
        st.markdown("Select an activity scenario to simulate:")
        
        # Scenario buttons with icons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🟢 NORMAL", key="normal", use_container_width=True):
                hr, acc = generate_sensor_data("NORMAL")
                st.session_state.heart_rate_raw = hr
                st.session_state.accelerometer_raw = acc
                st.session_state.scenario = "NORMAL"
                st.session_state.prediction = 0
                log_event("SCENARIO", "Normal activity selected")
                st.rerun()
        
        with col_btn2:
            if st.button("🟡 PLAYING", key="playing", use_container_width=True):
                hr, acc = generate_sensor_data("PLAYING")
                st.session_state.heart_rate_raw = hr
                st.session_state.accelerometer_raw = acc
                st.session_state.scenario = "PLAYING"
                st.session_state.prediction = 1
                log_event("SCENARIO", "Playing activity selected")
                st.rerun()
        
        with col_btn3:
            if st.button("🔴 DANGER", key="danger", use_container_width=True):
                hr, acc = generate_sensor_data("DANGER")
                st.session_state.heart_rate_raw = hr
                st.session_state.accelerometer_raw = acc
                st.session_state.scenario = "DANGER"
                st.session_state.prediction = 2
                log_event("SCENARIO", "Danger scenario selected")
                st.rerun()
        
        st.markdown("---")
        
        # Real-time vital signs monitor
        if st.session_state.heart_rate_raw and st.session_state.accelerometer_raw:
            st.markdown("### 📊 Real-time Vital Signs")
            
            # Vital signs card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {get_watch_color(st.session_state.prediction) if st.session_state.prediction is not None else '#667eea'} 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;
                        animation: {'pulse 0.5s infinite' if st.session_state.prediction == 2 else 'none'};">
            """, unsafe_allow_html=True)
            
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                current_hr = st.session_state.heart_rate_raw[-1] if st.session_state.heart_rate_raw else 0
                avg_hr = np.mean(st.session_state.heart_rate_raw)
                st.metric("Heart Rate (BPM)", f"{current_hr:.0f}", f"{avg_hr:.0f} avg")
            
            with col_metric2:
                current_acc = st.session_state.accelerometer_raw[-1] if st.session_state.accelerometer_raw else 0
                avg_acc = np.mean(st.session_state.accelerometer_raw)
                st.metric("Accelerometer (g)", f"{current_acc:.2f}", f"{avg_acc:.2f} avg")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Sensor data visualization
            st.markdown("### 📈 Sensor Data Stream")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                hr_df = pd.DataFrame({
                    'Heart Rate': st.session_state.heart_rate_raw[-20:] if len(st.session_state.heart_rate_raw) > 20 else st.session_state.heart_rate_raw
                })
                st.line_chart(hr_df, height=150)
            
            with col_chart2:
                acc_df = pd.DataFrame({
                    'Accelerometer': st.session_state.accelerometer_raw[-20:] if len(st.session_state.accelerometer_raw) > 20 else st.session_state.accelerometer_raw
                })
                st.line_chart(acc_df, height=150)
        else:
            # Placeholder
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; color: white; text-align: center;">
                <h3 style="color: white;">👆 Click a scenario button</h3>
                <p>Generate sensor data to begin monitoring</p>
            </div>
            """, unsafe_allow_html=True)
# ----------------------------------------------------------------------
# SECTION 9: AI ENGINE PAGE
# ----------------------------------------------------------------------

elif page == "AI Engine":
    st.title("AI Processing Engine")
    st.markdown("---")
    
    if not st.session_state.heart_rate_raw or not st.session_state.accelerometer_raw:
        st.warning("Please generate sensor data from the Smartwatch page first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Raw Sensor Data")
            
            df_raw = pd.DataFrame({
                'Heart Rate': st.session_state.heart_rate_raw,
                'Accelerometer': st.session_state.accelerometer_raw
            })
            
            st.line_chart(df_raw)
            
            st.markdown("### Data Summary")
            st.write(f"Heart Rate - Min: {min(st.session_state.heart_rate_raw):.1f}, "
                    f"Max: {max(st.session_state.heart_rate_raw):.1f}, "
                    f"Mean: {np.mean(st.session_state.heart_rate_raw):.1f}")
            st.write(f"Accelerometer - Min: {min(st.session_state.accelerometer_raw):.2f}, "
                    f"Max: {max(st.session_state.accelerometer_raw):.2f}, "
                    f"Mean: {np.mean(st.session_state.accelerometer_raw):.2f}")
        
        with col2:
            st.markdown("### Feature Engineering")
            
            features = compute_features(
                st.session_state.heart_rate_raw,
                st.session_state.accelerometer_raw
            )
            st.session_state.features = features
            
            st.markdown("""
            <div class="info-card">
                <h4>Calculated Features:</h4>
            """, unsafe_allow_html=True)
            
            col_feat1, col_feat2 = st.columns(2)
            with col_feat1:
                st.metric("HR Mean", f"{features['hr_mean']:.2f}")
                st.metric("HR Gradient", f"{features['hr_gradient']:.2f}")
            with col_feat2:
                st.metric("ACC Mean", f"{features['acc_mean']:.3f}")
                st.metric("ACC Variance", f"{features['acc_variance']:.3f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🚀 Run AI Model", type="primary", use_container_width=True):
                with st.spinner("Processing sensor data..."):
                    time.sleep(1)
                    
                    if model is not None:
                        feature_vector = np.array([[
                            features['hr_mean'],
                            features['hr_gradient'],
                            features['acc_mean'],
                            features['acc_variance']
                        ]])
                        
                        prediction = model.predict(feature_vector)[0]
                        probabilities = model.predict_proba(feature_vector)[0]
                        confidence = max(probabilities) * 100
                        
                        st.session_state.prediction = prediction
                        st.session_state.confidence = confidence
                        st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        log_event("AI_PROCESSING", f"Prediction: {get_prediction_label(prediction)}")
                        
                        st.rerun()
        
        if st.session_state.prediction is not None:
            st.markdown("---")
            st.markdown("### AI Model Results")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                pred_label = get_prediction_label(st.session_state.prediction)
                pred_color = get_watch_color(st.session_state.prediction)
                st.markdown(f"""
                <div style="background-color: {pred_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">{pred_label}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>Confidence</h4>
                    <p style="font-size: 2rem; font-weight: bold; color: {pred_color};">{st.session_state.confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res3:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>Processing Time</h4>
                    <p style="font-size: 1.5rem; font-weight: bold;">45 ms</p>
                    <p style="font-size: 0.9rem;">{st.session_state.timestamp}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Simulated GPS Location")
            st.info(f"📍 Location: {st.session_state.location} (Lat: 37.7749° N, Long: 122.4194° W)")

# ----------------------------------------------------------------------
# SECTION 10: NOTIFICATIONS PAGE
# ----------------------------------------------------------------------

elif page == "Notifications":
    st.title("Parent Notification System")
    st.markdown("---")
    
    if st.session_state.prediction is None:
        st.warning("No prediction available. Please run the AI Engine first.")
    else:
        if st.session_state.prediction == 0:
            st.markdown("""
            <div class="safe-banner">
                ✅ Child Status: SAFE - Normal Activity Detected
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        
        elif st.session_state.prediction == 1:
            st.markdown("""
            <div class="warning-banner">
                ⚡ Child Status: High Physical Activity - Playing
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.session_state.alert_triggered = True
            
            st.markdown("""
            <div class="danger-banner">
                🚨 HIGH RISK DETECTED - Immediate Attention Required 🚨
            </div>
            """, unsafe_allow_html=True)
            
            st.error("🔊 ALARM: Emergency Situation Detected!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Alert Details")
                st.markdown(f"""
                - **Heart Rate:** {st.session_state.features.get('hr_mean', 0):.1f} BPM
                - **Location:** {st.session_state.location}
                - **Confidence:** {st.session_state.confidence:.1f}%
                - **Timestamp:** {st.session_state.timestamp}
                - **Accelerometer Variance:** {st.session_state.features.get('acc_variance', 0):.3f}
                """)
            
            with col2:
                st.markdown("### Emergency Camera Feed")
                img_file = st.camera_input("Capture emergency photo", key="emergency_cam")
                
                if img_file is not None:
                    st.session_state.captured_image = img_file
                    st.image(img_file, caption="Emergency Photo Captured", use_container_width=True)
                    log_event("CAMERA", "Emergency photo captured")

# ----------------------------------------------------------------------
# SECTION 11: CONFIRMATION PAGE
# ----------------------------------------------------------------------

elif page == "Confirmation":
    st.title("Parent Confirmation")
    st.markdown("---")
    
    if st.session_state.prediction is None:
        st.warning("No alert to confirm. Please run the AI Engine first.")
    elif st.session_state.prediction != 2:
        st.info("No emergency action required. Current status is safe.")
    else:
        st.markdown("""
        <div class="danger-banner" style="animation: none;">
            🚨 EMERGENCY CONFIRMATION REQUIRED 🚨
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.captured_image:
            st.image(st.session_state.captured_image, caption="Captured Emergency Photo", width=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ CONFIRM EMERGENCY", type="primary", use_container_width=True):
                st.session_state.parent_confirmation = "CONFIRMED"
                log_event("CONFIRMATION", "Emergency confirmed by parent")
                st.success("Emergency Confirmed. School and emergency contacts notified.")
                st.balloons()
        
        with col2:
            if st.button("❌ FALSE ALARM", use_container_width=True):
                st.session_state.parent_confirmation = "FALSE_ALARM"
                log_event("CONFIRMATION", "Marked as false alarm by parent")
                st.warning("Event Marked as False Alarm. Feedback stored for model improvement.")
        
        if st.session_state.parent_confirmation:
            st.markdown("---")
            st.markdown("### Confirmation Status")
            if st.session_state.parent_confirmation == "CONFIRMED":
                st.markdown("""
                <div style="background-color: #10B981; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h4>✅ Emergency Protocol Activated</h4>
                    <p>School notified • Emergency contacts alerted • Location tracked</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #F59E0B; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h4>⚠️ False Alarm Registered</h4>
                    <p>Feedback stored for model retraining • No further action required</p>
                </div>
                """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SECTION 12: LOGS PAGE
# ----------------------------------------------------------------------

elif page == "Logs":
    st.title("System Event Logs")
    st.markdown("---")
    
    if not st.session_state.event_logs:
        st.info("No events logged yet. Start using the system to generate logs.")
    else:
        for log in reversed(st.session_state.event_logs):
            color = {
                "SCENARIO": "#1E3A8A",
                "AI_PROCESSING": "#059669",
                "CAMERA": "#7C3AED",
                "CONFIRMATION": "#D97706"
            }.get(log['type'], "#6B7280")
            
            st.markdown(f"""
            <div style="background-color: #F9FAFB; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {color};">
                <strong>[{log['timestamp']}]</strong> {log['type']}: {log['details']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", len(st.session_state.event_logs))
        with col2:
            scenarios = sum(1 for log in st.session_state.event_logs if log['type'] == "SCENARIO")
            st.metric("Scenarios Run", scenarios)
        with col3:
            alerts = sum(1 for log in st.session_state.event_logs if 'DANGER' in str(log['details']))
            st.metric("Danger Alerts", alerts)
        with col4:
            confirmations = sum(1 for log in st.session_state.event_logs if log['type'] == "CONFIRMATION")
            st.metric("Confirmations", confirmations)
        
        if st.button("🗑️ Clear All Logs"):
            st.session_state.event_logs = []
            st.rerun()

# ----------------------------------------------------------------------
# SECTION 13: FOOTER
# ----------------------------------------------------------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 20px;">
    <p>🛡️ SWIRI AI Safety System v1.0 | Real-time Child Safety Monitoring</p>
    <p style="font-size: 0.8rem;">© 2024 SWIRI Technologies | All sensor data is simulated for demonstration</p>
</div>
""", unsafe_allow_html=True)