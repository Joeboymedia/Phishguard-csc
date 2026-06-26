import streamlit as st
import joblib
import pandas as pd
from feature_extractor import extract_features

# --- 1. Setup & Configuration ---
st.set_page_config(page_title="PhishGuard Dual Engine", page_icon="🛡️", layout="wide")

# Feature names for the URL ML Model to prevent warnings
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
    # Hardcoded credentials for project demonstration
    if username == "admin" and password == "admin123":
        st.sidebar.success("Logged in successfully!")
        st.sidebar.info("System Logs: \n- Text Scan 1: Safe\n- URL Scan 1: Phishing\n*(Database connection goes here)*")
    else:
        st.sidebar.error("Invalid credentials")

# --- 3. Load All AI Models ---
@st.cache_resource
def load_all_models():
    try:
        # NLP Models (for Emails/SMS)
        nlp_model = joblib.load("models/nlp_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        # ML Model (for URLs)
        url_model = joblib.load("models/best_model.pkl")
        return nlp_model, vectorizer, url_model
    except Exception as e:
        st.error(f"Failed to load AI files. Error: {e}")
        return None, None, None

nlp_model, vectorizer, url_model = load_all_models()

# Mini NLP scanner specifically for catching bad words inside URLs
def url_nlp_scanner(url):
    suspicious_words = ['verify', 'login', 'update', 'bank', 'secure', 'confirm', 'account', 'paypal']
    url_lower = url.lower()
    return [word for word in suspicious_words if word in url_lower]

# --- 4. Main App UI ---
st.title("🛡️ PhishGuard: Dual-Engine Detection System")
st.markdown("Choose the type of content you want to scan using the tabs below.")

# Create the Tabs
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
                
                st.divider()
                if prediction == 0:
                    st.error("### 🚨 VERDICT: PHISHING DETECTED")
                    st.markdown("This message contains linguistic patterns highly consistent with social engineering and fraud.")
                else:
                    st.success("### ✅ VERDICT: SAFE")
                    st.markdown("No significant phishing indicators were detected in the text of this message.")

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
                # 1. NLP Keyword check for URL
                found_words = url_nlp_scanner(url_input)
                nlp_flag = len(found_words) > 0
                
                # 2. Mathematical feature extraction
                features = extract_features(url_input)
                features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
                
                # Check critical structural flaws directly (Feature 0 is the IP Address check)
                critical_structural_flaw = (features[0][0] == -1) 
                
                ml_result = url_model.predict(features_df)[0]
                ml_flag = (ml_result != 1) 
                
                st.divider()
                
                # If the ML flags it, the NLP flags it, OR a critical flaw (like an IP) is found
                if ml_flag or nlp_flag or critical_structural_flaw:
                    st.error("### 🚨 VERDICT: PHISHING DETECTED")
                    st.markdown("This URL is highly suspicious and should be avoided.")
                    with st.expander("See technical details"):
                        if ml_flag:
                            st.write("- **Machine Learning Flag:** The URL structure matches known phishing patterns.")
                        if critical_structural_flaw:
                            st.write("- **Structural Flag:** Use of a direct IP Address instead of a standard domain.")
                        if nlp_flag:
                            st.write(f"- **NLP Flag:** Found suspicious keywords: `{', '.join(found_words)}`")
                else:
                    st.success("### ✅ VERDICT: SAFE")
                    st.markdown("No significant threats were detected in this URL by either the ML model or the NLP scanner.")