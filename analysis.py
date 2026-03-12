import requests
from bs4 import BeautifulSoup

url = "https://db.netkeiba.com/horse/2023107321"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

res = requests.get(url, headers=headers)

soup = BeautifulSoup(res.text, "html.parser")

tables = soup.find_all("table")

print("テーブル数:", len(tables))

for t in tables:
    print(t.get("class"))