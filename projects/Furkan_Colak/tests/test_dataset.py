from src.dataset import clean, make_splits, normalize


def test_splits_do_not_share_sources():
    ids = [f"s{i}" for i in range(100)] * 3
    splits = make_splits(ids, [42, 123, 2024], {"train": 0.7, "val": 0.1, "test": 0.2})
    for s in splits.values():
        tr, va, te = map(set, (s["train"], s["val"], s["test"]))
        assert not (tr & te) and not (tr & va) and not (va & te)
        assert len(tr | va | te) == 100


def test_splits_are_deterministic_and_differ_by_seed():
    ids = [f"s{i}" for i in range(50)]
    a = make_splits(ids, [42, 123], {"train": 0.7, "val": 0.1, "test": 0.2})
    b = make_splits(ids, [42, 123], {"train": 0.7, "val": 0.1, "test": 0.2})
    assert a == b
    assert a["42"]["test"] != a["123"]["test"]


def test_clean_removes_llm_wrapping():
    raw = (
        "<think>plan the answer</think>Here is the bulletin:\n"
        "## Threat Summary\n"
        "**APT29** used spear‑phishing to deliver a loader.\n"
        "- It targeted defense contractors.\n"
        "(Word count: 21)"
    )
    out = clean(raw)
    assert out == "APT29 used spear-phishing to deliver a loader. It targeted defense contractors."


def test_normalize_keeps_human_text():
    text = "The actors exploited CVE-2021-44228 in VMware Horizon. They installed XMRig."
    assert normalize(text) == text
    assert clean(text) == text


def test_clean_drops_plain_title_line():
    raw = ("Incident Report: Iranian Government-Sponsored APT Actors Compromise FCEB Organization\n\n"
           "The actors exploited Log4Shell.")
    assert clean(raw) == "The actors exploited Log4Shell."


def test_chunk_respects_word_limits():
    from src.collect import chunk
    p = lambda n: " ".join(["w"] * n)
    out = chunk([p(100), p(100), p(900), p(120), p(60), p(200)], 150, 350, 5)
    assert [len(c.split()) for c in out] == [200, 180, 200]


def test_footnotes_removed():
    from src.collect import FOOTNOTE
    assert FOOTNOTE.sub(" ", "exposed to the internet. [ 1 ] Typically, [2, 3] it").split() == \
        "exposed to the internet. Typically, it".split()


def test_footnote_ranges_removed():
    from src.collect import FOOTNOTE
    assert FOOTNOTE.sub(" ", "the European Union. [1-5] This report").split() == \
        "the European Union. This report".split()


def test_clean_drops_memo_headers_and_labels():
    raw = ("**Internal Threat Bulletin**\n\n**Subject:** Iran-based Threat Actor Activity Detected\n\n"
           "**Summary:**\n\nOur team found activity. **Impact:** It may spread.\n\n"
           "Here is the revised draft:\n\nThe actor used VPN access.")
    assert clean(raw) == "Our team found activity. Impact: It may spread. The actor used VPN access."


def test_agency_names_masked():
    from src.dataset import AGENCIES
    t = "CISA and the FBI observed activity; the Federal Bureau of Investigation and NSA agree."
    assert AGENCIES.sub("AGENCY", t) == "AGENCY and the AGENCY observed activity; the AGENCY and AGENCY agree."
