import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL verification warning
urllib3.disable_warnings(InsecureRequestWarning)

# Make the request without SSL verification
response = requests.get('https://api.groq.com', verify=False)
print(response.status_code)


