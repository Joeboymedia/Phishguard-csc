import re
from urllib.parse import urlparse
import ipaddress

def has_ip(url):
    try:
        host = urlparse(url).netloc
        ipaddress.ip_address(host.split(':')[0])
        return -1  # -1 means Phishing (IP Address detected)
    except:
        return 1   # 1 means Safe (Standard domain name detected)

def url_length(url):
    return 1 if len(url) < 54 else -1

def shortening_service(url):
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
    return -1 if any(s in url for s in shorteners) else 1

def at_symbol(url):
    return -1 if '@' in url else 1

def double_slash(url):
    pos = url.rfind('//')
    return -1 if pos > 7 else 1

def prefix_suffix(url):
    domain = urlparse(url).netloc
    return -1 if '-' in domain else 1

def sub_domain(url):
    domain = urlparse(url).netloc
    dots = domain.count('.')
    if dots == 1:
        return 1
    return -1

def https_token(url):
    domain = urlparse(url).netloc
    return -1 if 'https' in domain.lower() else 1

def extract_features(url):
    # The exact 30 features in the exact order your model requires
    features = [
        has_ip(url),               # 1. having_IP_Address
        url_length(url),           # 2. URL_Length
        shortening_service(url),   # 3. Shortining_Service
        at_symbol(url),            # 4. having_At_Symbol
        double_slash(url),         # 5. double_slash_redirecting
        prefix_suffix(url),        # 6. Prefix_Suffix
        sub_domain(url),           # 7. having_Sub_Domain
        1,                         # 8. SSLfinal_State (Placeholder)
        1,                         # 9. Domain_registeration_length (Placeholder)
        1,                         # 10. Favicon (Placeholder)
        1,                         # 11. port (Placeholder)
        https_token(url),          # 12. HTTPS_token
        1,                         # 13. Request_URL (Placeholder)
        1,                         # 14. URL_of_Anchor (Placeholder)
        1,                         # 15. Links_in_tags (Placeholder)
        1,                         # 16. SFH (Placeholder)
        1,                         # 17. Submitting_to_email (Placeholder)
        1,                         # 18. Abnormal_URL (Placeholder)
        1,                         # 19. Redirect (Placeholder)
        1,                         # 20. on_mouseover (Placeholder)
        1,                         # 21. RightClick (Placeholder)
        1,                         # 22. popUpWidnow (Placeholder)
        1,                         # 23. Iframe (Placeholder)
        1,                         # 24. age_of_domain (Placeholder)
        1,                         # 25. DNSRecord (Placeholder)
        1,                         # 26. web_traffic (Placeholder)
        1,                         # 27. Page_Rank (Placeholder)
        1,                         # 28. Google_Index (Placeholder)
        1,                         # 29. Links_pointing_to_page (Placeholder)
        1                          # 30. Statistical_report (Placeholder)
    ]
    
    # Models expect a 2D array (a list inside a list) for a single prediction
    return [features]

if __name__ == '__main__':
    url = input('Enter URL: ')
    print("Extracted 30 Features Array:")
    print(extract_features(url))