import streamlit as st
import joblib
import pandas as pd
from feature_extractor import extract_features
from datetime import datetime

# --- 1. Setup & Configuration ---
st.set_page_config(page_title="PhishGuard Dual Engine", page_icon="🛡️", layout="wide")

# Initialize Live System Memory for Logs
if 'scan_logs' not in st.session_state:
    st.session_state.scan_logs = []

FEATURE_NAMES = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 
    'Statistical_report'
]

# --- 2. Admin Login Sidebar ---
st.sidebar.title("Admin Dashboard")
st.sidebar.markdown("- - - Administrator Login - - -")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if username == "admin" and password == "admin123":
        st.sidebar.success("Logged in successfully!")
        st.sidebar.markdown("### 📡 Live System Logs")
        
        # Display the real logs from memory
        if len(st.session_state.scan_logs) == 0:
            st.sidebar.info("No scans have been performed in this session yet.")
        else:
            # Show the most recent scans at the top
            for log in reversed(st.session_state.scan_logs):
                if "PHISHING" in log:
                    st.sidebar.error(log)
                else:
                    st.sidebar.success(log)
    else:
        st.sidebar.error("Invalid credentials")

# --- 3. Load All AI Models ---
@st.cache_resource
def load_all_models():
    try:
        # Added "models/" back to the beginning of these three lines!
        nlp_model = joblib.load("models/nlp_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        url_model = joblib.load("models/best_model.pkl")
        return nlp_model, vectorizer, url_model
    except Exception as e:
        st.error(f"Failed to load AI files. Error: {e}")
        return None, None, None
# --- 4. Main App UI ---
st.title("🛡️ PhishGuard: Dual-Engine Detection System")
st.markdown("Choose the type of content you want to scan using the tabs below.")

tab1, tab2 = st.tabs(["📧 Email & SMS Scanner (NLP)", "🌐 URL Scanner (ML + Hybrid)"])

# ====== TAB 1: TEXT SCANNER ======
with tab1:
    st.subheader("Text-Based Social Engineering Detection")
    user_text = st.text_area("Message Content:", height=150, placeholder="Paste the suspicious email or text message here...")
    
    if st.button("Scan Message"):
        if not user_text.strip():
            st.warning("Please paste some text to analyze.")
        elif nlp_model is None:
            st.error("NLP engine is offline.")
        else:
            with st.spinner("Analyzing linguistic patterns..."):
                text_features = vectorizer.transform([user_text])
                prediction = nlp_model.predict(text_features)[0]
                time_now = datetime.now().strftime("%H:%M:%S")
                
                st.divider()
                if prediction == 0:
                    st.error("### 🚨 VERDICT: PHISHING DETECTED")
                    st.markdown("This message contains linguistic patterns highly consistent with social engineering and fraud.")
                    # Save to Live Log
                    st.session_state.scan_logs.append(f"[{time_now}] Text Scan: PHISHING")
                else:
                    st.success("### ✅ VERDICT: SAFE")
                    st.markdown("No significant phishing indicators were detected in the text of this message.")
                    # Save to Live Log
                    st.session_state.scan_logs.append(f"[{time_now}] Text Scan: SAFE")

# ====== TAB 2: URL SCANNER ======
with tab2:
    st.subheader("URL Structural & Hybrid Detection")
    url_input = st.text_input("Enter a URL to scan:", placeholder="e.g., https://secure-login-update.com")
    
    if st.button("Scan URL"):
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        elif url_model is None:
            st.error("URL ML engine is offline.")
        else:
            with st.spinner("Extracting structural features..."):
                found_words = url_nlp_scanner(url_input)
                nlp_flag = len(found_words) > 0
                
                features = extract_features(url_input)
                features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
                
                critical_structural_flaw = (features[0][0] == -1) 
                
                ml_result = url_model.predict(features_df)[0]
                ml_flag = (ml_result != 1) 
                time_now = datetime.now().strftime("%H:%M:%S")
                
                st.divider()
                if ml_flag or nlp_flag or critical_structural_flaw:
                    st.error("### 🚨 VERDICT: PHISHING DETECTED")
                    st.markdown("This URL is highly suspicious and should be avoided.")
                    # Save to Live Log
                    st.session_state.scan_logs.append(f"[{time_now}] URL Scan ({url_input[:20]}...): PHISHING")
                else:
                    st.success("### ✅ VERDICT: SAFE")
                    st.markdown("No significant threats were detected in this URL by either the ML model or the NLP scanner.")
                    # Save to Live Log
                    st.session_state.scan_logs.append(f"[{time_now}] URL Scan ({url_input[:20]}...): SAFE")
