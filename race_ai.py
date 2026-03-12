from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# レースURLを入力
race_url = input("レースURLを貼ってください: ")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(race_url)

time.sleep(3)

# レース条件取得
race_info = driver.find_element(By.CLASS_NAME, "RaceData01").text

# 距離
if "1600" in race_info:
    TARGET_DISTANCE = "1600"
elif "1800" in race_info:
    TARGET_DISTANCE = "1800"
elif "2000" in race_info:
    TARGET_DISTANCE = "2000"
else:
    TARGET_DISTANCE = ""

# コース
if "中山" in race_info:
    TARGET_COURSE = "中山"
elif "東京" in race_info:
    TARGET_COURSE = "東京"
elif "京都" in race_info:
    TARGET_COURSE = "京都"
elif "阪神" in race_info:
    TARGET_COURSE = "阪神"
else:
    TARGET_COURSE = ""

print("今回のレース条件")
print("距離:", TARGET_DISTANCE)
print("コース:", TARGET_COURSE)

# 出走馬取得
horse_elements = driver.find_elements(By.CSS_SELECTOR, ".HorseName a")

horses = []

for h in horse_elements:
    horses.append({
        "name": h.text,
        "link": h.get_attribute("href")
    })

results = []

print("\n===== 出走馬分析 =====")

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
        distance = cols[14].text

        print(date, course, distance, rank_text)

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

        # 距離適性
        if TARGET_DISTANCE in distance:
            total_score += 2

        # コース適性
        if TARGET_COURSE in course:
            total_score += 1

    print("AIスコア:", total_score)

    results.append({
        "name": horse["name"],
        "score": total_score
    })

driver.quit()

print("\n===== AIランキング =====")

results = sorted(results, key=lambda x: x["score"], reverse=True)

for i, r in enumerate(results, start=1):
    print(f"{i}位 {r['name']}  score={r['score']}")