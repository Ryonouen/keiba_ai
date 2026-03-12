from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

url = "https://race.netkeiba.com/race/shutuba.html?race_id=202406020611"

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)

time.sleep(3)

horses = driver.find_elements(By.CSS_SELECTOR, ".HorseName a")

print("出走馬")

for h in horses:
    name = h.text
    link = h.get_attribute("href")
    
    print(name, link)

driver.quit()