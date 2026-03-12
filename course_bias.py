import requests
from bs4 import BeautifulSoup
import pandas as pd


# netkeiba競馬場ID
COURSE_ID_MAP = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}


def _normalize_bias(bias_dict):
    """
    出現率 → AI倍率に変換
    平均1.0になるように補正
    """
    if not bias_dict:
        return {"front": 1.0, "stalker": 1.0, "closer": 1.0}

    total = sum(bias_dict.values())

    if total == 0:
        return {"front": 1.0, "stalker": 1.0, "closer": 1.0}

    avg = total / len(bias_dict)

    return {
        "front": bias_dict.get("front", avg) / avg,
        "stalker": bias_dict.get("stalker", avg) / avg,
        "closer": bias_dict.get("closer", avg) / avg,
    }


def get_course_bias(course, surface, distance):
    """
    競馬場 × 距離 の脚質バイアス取得

    return例
    {
        "front":1.08,
        "stalker":1.02,
        "closer":0.93
    }
    """

    track = COURSE_ID_MAP.get(course)

    if track is None:
        return {"front": 1.0, "stalker": 1.0, "closer": 1.0}

    url = f"https://db.netkeiba.com/?pid=race_list&track={track}&kyori={distance}&type={surface}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        races = soup.select("table tr")

        styles = []

        for race in races:

            cols = race.find_all("td")

            if len(cols) < 8:
                continue

            passing = cols[7].text.strip()

            if "-" not in passing:
                continue

            try:
                first = int(passing.split("-")[0])
            except Exception:
                continue

            if first <= 2:
                styles.append("front")
            elif first <= 5:
                styles.append("stalker")
            else:
                styles.append("closer")

        if not styles:
            return {"front": 1.0, "stalker": 1.0, "closer": 1.0}

        df = pd.Series(styles)

        raw_bias = df.value_counts(normalize=True).to_dict()

        return _normalize_bias(raw_bias)

    except Exception:

        return {"front": 1.0, "stalker": 1.0, "closer": 1.0}
