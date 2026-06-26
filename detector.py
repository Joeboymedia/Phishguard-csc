import joblib
import pandas as pd
from feature_extractor import extract_features

MODEL_PATH = "models/best_model.pkl"

FEATURE_NAMES = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 
    'Statistical_report'
]

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"❌ Error: Could not find the model file at {MODEL_PATH}.")
    exit()

def nlp_scanner(url):
    """Scans the URL for suspicious keywords used in social engineering."""
    suspicious_words = ['verify', 'login', 'update', 'bank', 'secure', 'confirm', 'account', 'paypal']
    found_words = []
    
    url_lower = url.lower()
    for word in suspicious_words:
        if word in url_lower:
            found_words.append(word)
            
    return found_words

def test_url(url):
    print(f"\nAnalyzing: {url}...")
    
    # --- 1. Run the NLP Scanner ---
    found_words = nlp_scanner(url)
    nlp_flag = len(found_words) > 0
    
    # --- 2. Run the Machine Learning Model ---
    features = extract_features(url)
    features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    
    prediction = model.predict(features_df)
    ml_result = prediction[0]
    ml_flag = (ml_result != 1) # If it's not 1 (Safe), the ML flagged it
    
    # --- 3. Debug Output ---
    print(f"[Debug] ML output: {ml_result} | NLP words found: {found_words}")
    
    # --- 4. The Hybrid Verdict ---
    # If EITHER the ML model catches it OR the NLP catches it, block the site.
    if ml_flag or nlp_flag:
        print("🚨 VERDICT: This looks like a PHISHING site!")
        
        # Explain exactly why it was blocked
        print("   Flags triggered:")
        if ml_flag:
            print("   - Machine Learning: Structural anomalies detected in the URL.")
        if nlp_flag:
            print(f"   - NLP Scanner: Found suspicious social engineering keywords: {', '.join(found_words)}")
    else:
        print("✅ VERDICT: This site appears to be SAFE.")

if __name__ == '__main__':
    print("--- Hybrid Phishing Detection System Active ---")
    user_url = input("Enter a URL to scan: ")
    test_url(user_url)