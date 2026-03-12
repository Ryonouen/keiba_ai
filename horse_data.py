from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

url = "https://db.netkeiba.com/horse/2023107321"

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get(url)

time.sleep(3)

rows = driver.find_elements(By.CSS_SELECTOR, ".db_h_race_results tbody tr")

print("日付 | 競馬場 | 距離 | 着順")

for r in rows[:5]:

    cols = r.find_elements(By.TAG_NAME, "td")

    date = cols[0].text
    course = cols[1].text
    rank = cols[11].text
    distance = cols[14].text

    print(date, course, distance, rank)

driver.quit()