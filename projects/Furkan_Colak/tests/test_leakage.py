import pandas as pd

from src.dedup import containment, dup_groups, shingles

REPOST = "The actors exploited a VPN flaw, moved laterally with stolen credentials and deployed ransomware on servers."
OTHER = "Phishing emails carried a macro document that dropped a loader and contacted a command server every hour."


def test_containment_finds_the_copy_and_ignores_unrelated_text():
    best, arg = containment([shingles(REPOST, 5)], [shingles(OTHER, 5), shingles(REPOST + " Patch now.", 5)])
    assert best[0] == 1.0 and arg[0] == 1
    best, _ = containment([shingles(OTHER, 5)], [shingles(REPOST, 5)])
    assert best[0] == 0.0


def test_reposted_reports_share_a_group():
    human = pd.DataFrame({"source_id": ["aa22-110a", "russian-threats", "other"], "text": [REPOST, REPOST, OTHER]})
    g = dup_groups(human, 5, 0.5)
    assert g["aa22-110a"] == g["russian-threats"] != g["other"]

