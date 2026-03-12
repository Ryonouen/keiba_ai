import requests
from bs4 import BeautifulSoup

url = "https://race.netkeiba.com/race/shutuba.html?race_id=202606020611"

headers = {
    "User-Agent": "Mozilla/5.0"
}

res = requests.get(url, headers=headers, verify=False)
res.encoding = "euc-jp"

soup = BeautifulSoup(res.text, "html.parser")

rows = soup.select("table.Shutuba_Table tr")

print("馬名 | 馬ページURL")

for row in rows:

    horse = row.select_one(".HorseName a")

    if horse:

        name = horse.text.strip()
        link = horse["href"]

        print(name, link)