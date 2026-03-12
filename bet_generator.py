import itertools

def generate_trio_box(features):

    features.sort(
        key=lambda x: x["win_prob"],
        reverse=True
    )

    top5 = features[:5]

    horses = [h["horse_name"] for h in top5]

    bets = []

    for combo in itertools.combinations(horses, 3):

        bets.append({
            "type": "trio",
            "horses": combo,
            "bet": 100
        })

    return bets


def generate_trifecta_ai(features):

    features.sort(
        key=lambda x: x["win_prob"],
        reverse=True
    )

    top5 = features[:5]
    first = top5[:2]

    bets = []

    for f in first:
        for s in top5:
            for t in top5:

                if len({
                    f["horse_name"],
                    s["horse_name"],
                    t["horse_name"]
                }) == 3:

                    bets.append({
                        "type": "trifecta",
                        "horses": (
                            f["horse_name"],
                            s["horse_name"],
                            t["horse_name"]
                        ),
                        "bet": 100
                    })

    return bets


def generate_ai_bets(features):
    """
    Main entry point for AI bet generation.
    Combines trio box (stable) and trifecta AI (high return).
    """

    # Copy list so original order is not destroyed outside
    features_sorted = sorted(
        features,
        key=lambda x: x["win_prob"],
        reverse=True
    )

    trio_bets = generate_trio_box(features_sorted)
    trifecta_bets = generate_trifecta_ai(features_sorted)

    return {
        "trio_box": trio_bets,
        "trifecta": trifecta_bets
    }