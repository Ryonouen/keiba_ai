from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

race_url = "https://race.netkeiba.com/race/shutuba.html?race_id=202406020611"

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(race_url)

time.sleep(3)

# 最初に出走馬の名前とURLだけ保存する
horse_elements = driver.find_elements(By.CSS_SELECTOR, ".HorseName a")

horses = []

for h in horse_elements:
    horses.append({
        "name": h.text,
        "link": h.get_attribute("href")
    })

print("===== 出走馬分析 =====")

results = []

for horse in horses:
    print("\n", horse["name"])

    driver.get(horse["link"])
    time.sleep(2)

    rows = driver.find_elements(By.CSS_SELECTOR, ".db_h_race_results tbody tr")

    total_score = 0

    for r in rows[:5]:
        cols = r.find_elements(By.TAG_NAME, "td")

        date = cols[0].text
        course = cols[1].text
        rank_text = cols[11].text

        print(date, course, rank_text)

        # 着順が数字のときだけ点数化する
        if rank_text.isdigit():
            rank = int(rank_text)

            if rank == 1:
                score = 5
            elif rank == 2:
                score = 4
            elif rank == 3:
                score = 3
            elif rank <= 5:
                score = 2
            elif rank <= 9:
                score = 1
            else:
                score = 0

            total_score += score

    results.append({
        "name": horse["name"],
        "score": total_score
    })

driver.quit()

print("\n===== AIランキング =====")

results = sorted(results, key=lambda x: x["score"], reverse=True)

for i, r in enumerate(results, start=1):
    print(f"{i}位 {r['name']}  score={r['score']}")