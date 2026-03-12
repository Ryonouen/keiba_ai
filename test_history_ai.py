from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from race_history_ai import (
    fetch_race_history,
    analyze_race_trend,
    build_race_trend_summary,
    build_winner_condition_ai,
    analyze_with_chatgpt
)

# テストレース
race_id = "202406050811"  # スプリングSなど

# Selenium起動
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

print("過去レース取得中...")

history = fetch_race_history(driver, race_id)

print("取得データ")
print(history)

print("\nレース傾向")
trend = analyze_race_trend(history)
print(trend)

print("\nレース特徴まとめ")
summary = build_race_trend_summary(history)
print(summary)

print("\n勝ち馬共通条件")
conditions = build_winner_condition_ai(history)
print(conditions)

print("\nChatGPT分析")
gpt = analyze_with_chatgpt(history, "スプリングS")
print(gpt)

driver.quit()