import requests
from bs4 import BeautifulSoup
import re

def get_contributions(username):
    url = f"https://github.com/{username}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, 'html.parser')
    h2 = soup.find('h2', class_='f4 text-normal mb-2')
    if h2:
        text = h2.get_text(strip=True)
        print("Found text:", text)
        match = re.search(r'([\d,]+)\s+contributions', text)
        if match:
            return int(match.group(1).replace(',', ''))
    return None

print("Torvalds:", get_contributions("torvalds"))
