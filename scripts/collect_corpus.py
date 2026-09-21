"""W1 코퍼스 수집 워커 — MediaWiki API 수집 (HTML 크롤링 없음).

GUIDE.md §3 준수:
- 한국어 우선, 부족분 영문 보충 (총 60~80건, 한국어 최소 25건)
- 2홉 후보 = 시드 본문 링크(prop=links) 중 여러 시드가 함께 가리키는 순
- 시드와 분류 1개 이상 공유한 것만 채택, 연도/목록/틀/분류 문서 제외
- 본문 prop=extracts + explaintext, 800자 미만 토막글 제외
- 표준 라이브러리 + requests만 사용, LLM 호출 없음
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

# ---------------- 시드 (GUIDE.md §3 그대로) ----------------
SEEDS_KO = [
    "바우하우스",
    "발터 그로피우스",
    "루트비히 미스 반 데어 로에",
    "르 코르뷔지에",
    "마르셀 브로이어",
    "데 스테일",
    "미술공예운동",
    "디터 람스",
    "울름 조형대학",
    "찰스 임스",
]

SEEDS_EN = [
    "Bauhaus",
    "Walter Gropius",
    "Ludwig Mies van der Rohe",
    "Marcel Breuer",
    "László Moholy-Nagy",
    "Josef Albers",
    "De Stijl",
    "Dieter Rams",
    "Ulm School of Design",
    "Herman Miller",
    "Knoll",
    "Vitra",
    "Charles and Ray Eames",
    "Massimo Vignelli",
    "Paul Rand",
]

API_KO = "https://ko.wikipedia.org/w/api.php"
API_EN = "https://en.wikipedia.org/w/api.php"

# ASCII only (한글 넣으면 인코딩 오류)
USER_AGENT = "DesignHistory-GraphRAG-W1/1.0 (research project; contact: admin@example.com)"

MIN_CHARS = 800
TARGET_TOTAL_MIN = 60
TARGET_TOTAL_MAX = 80
TARGET_KO_MIN = 25
TARGET_TOTAL_AIM = 70  # 60~80 중간값
SLEEP = 0.3

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.json"

NS_PREFIXES = (
    "분류:", "틀:", "위키백과:", "파일:", "포털:", "초안:", "도움말:", "사용자:", "토론:",
    "Category:", "Template:", "Wikipedia:", "File:", "Portal:", "Draft:", "Help:",
    "User:", "Talk:", "MediaWiki:", "Module:", "모듈:",
)

TITLE_EXCLUDE_RES = [
    re.compile(r"^\d{3,4}년"),
    re.compile(r"\d{4}년"),
    re.compile(r"^\d{4}\s?(in|년|-)"),
    re.compile(r"^List of\b", re.IGNORECASE),
    re.compile(r"\b(List of|Timeline of|Outline of)\b", re.IGNORECASE),
    re.compile(r"목록$"),
    re.compile(r" 목록"),
    re.compile(r"(연표|일람|동음이의)"),
    re.compile(r"^\d{1,4}년대"),
    re.compile(r"^Category:.*(years|lists|templates)", re.IGNORECASE),
]

# 주제 분류가 아닌 추적/관리용 분류. ko 위키는 hidden 표시가 없어
# clshow=!hidden 으로도 걸러지지 않으므로 별도로 제외한다.
# (예: '독일어 표기를 포함한 문서' 하나 때문에 무솔리니 같은 문서가
#  시드와 '공유' 판정이 나는 가짜 다리를 차단. GUIDE §3 탈락 기준
#  '스키마 5종에 안 맞는 문서'에 대응, manifest 사유는 '기타'로 기록)
TRACKING_CAT_RES = [
    re.compile(r"표기를 포함한 문서"),
    re.compile(r"위키데이터"),
    re.compile(r"위키공용|공용분류|Commons"),
    re.compile(r"출처가 필요"),
    re.compile(r"토막글"),
    re.compile(r"인용한 문서"),
    re.compile(r"오류가 있는 문서"),
    re.compile(r"해결되지 않은 속성"),
    re.compile(r"로컬에는 없지만"),
    re.compile(r"표준 도서 번호"),
    re.compile(r"Harv|Sfn"),
    re.compile(r"Articles (containing|with|needing|lacking)"),
    re.compile(r"^(Use |All |Pages with|Wikipedia|Webarchive|CS1|AC with|Good articles|Featured)"),
]


# 주제 다리가 아니라 가짜 다리를 만드는 허브형 분류. GUIDE §2에서
# BORN_IN/Died_IN·지역 노드를 버린 것과 같은 이유(차수 폭발 → 의미 없는 경로).
# 출생연도·혈통·출신지·사인·훈장·기념물(지폐 인물 등)·종교·군경력·정치직은
# 직업·사조·학교 같은 주제 연결이 아니므로 공유 판정에서 제외한다.
# ('헝가리 유대인'처럼 '계 X인' 패턴이 아닌 것은 남긴다 — 브로이어↔모홀리나지
#  같은 바우하우스 망명자 연결을 끊지 않기 위함. manifest의 shared로 감사 가능)
HUB_CAT_RES = [
    re.compile(r"^\d{3,4}년 (출생|사망)$"),
    re.compile(r"^\d{3,4}s? (births|deaths)$"),
    re.compile(r"계 .+인$"),
    re.compile(r"이민간 사람$"),
    re.compile(r"출신$"),
    re.compile(r"[으]로 죽은 사람$"),
    re.compile(r"수훈자$"),
    re.compile(r"의 인물$"),
    re.compile(r"교도$"),
    re.compile(r"참전 군인$"),
    re.compile(r"(정치인|대통령|부통령|장관|의원|후보|장군|총리)$"),
    re.compile(r"descent$"),
    re.compile(r"(?i)^(German|Italian|Dutch|Hungarian|Swiss|Austrian|French|Russian|British|Jewish) (Americans|people|Jews|emigrants|immigrants)"),
    re.compile(r"(?i)(immigrants? to|emigrants? from)"),
    re.compile(r"(?i)(banknotes?|postage stamps?|depictions of)"),
    re.compile(r"(?i)deaths? (from|due to|by)"),
    re.compile(r"(?i)recipients of"),
    re.compile(r"(?i)(military personnel|war veterans|Senators|Presidents|politicians)"),
    re.compile(r"(?i)^People from "),
]


def is_tracking_cat(cat: str) -> bool:
    return any(rx.search(cat) for rx in TRACKING_CAT_RES)


def is_hub_cat(cat: str) -> bool:
    return any(rx.search(cat) for rx in HUB_CAT_RES)


def topical(cats: list) -> list:
    # 분류명은 '분류:'/'Category:' 접두사가 붙어 오므로 접두사를 뗀 형태로 판정한다
    # (^ 앵커 패턴이 동작하도록). 반환은 원본 그대로(비교 일관성 유지).
    return [
        c
        for c in cats
        if not is_tracking_cat(short_category(c)) and not is_hub_cat(short_category(c))
    ]


def is_excluded_title(title: str) -> bool:
    for p in NS_PREFIXES:
        if title.startswith(p):
            return True
    for rx in TITLE_EXCLUDE_RES:
        if rx.search(title):
            return True
    return False


def api_query(endpoint: str, params: dict) -> dict:
    """continue를 끝까지 이어받아 합친 query 결과를 반환한다."""
    params = dict(params)
    params["format"] = "json"
    params["formatversion"] = "2"
    merged_pages = None
    merged_redirects: list = []
    merged_dict = None
    first = True
    while True:
        r = requests.get(endpoint, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "query" in data:
            merged_redirects.extend(data["query"].get("redirects", []))
        if "query" in data and "pages" in data["query"]:
            if first:
                merged_dict = data
                merged_pages = {}
                for p in data["query"]["pages"]:
                    key = p.get("pageid", p.get("title"))
                    merged_pages[key] = p
                first = False
            else:
                for p in data["query"]["pages"]:
                    key = p.get("pageid", p.get("title"))
                    if key in merged_pages:
                        for k in ("links", "categories"):
                            if k in p:
                                merged_pages[key].setdefault(k, []).extend(p[k])
                    else:
                        merged_pages[key] = p
        elif first:
            merged_dict = data
            first = False
        time.sleep(SLEEP)
        if "continue" not in data:
            break
        params.update(data["continue"])
    if merged_dict is not None and "query" in merged_dict and merged_pages is not None:
        merged_dict["query"]["pages"] = list(merged_pages.values())
        merged_dict["query"]["redirects"] = merged_redirects
        return merged_dict
    return merged_dict if merged_dict is not None else {}


def get_categories(endpoint: str, titles: list) -> dict:
    """제목 -> 분류명 리스트. 20건씩 묶어 호출."""
    out: dict[str, list] = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i : i + 20]
        data = api_query(
            endpoint,
            {
                "action": "query",
                "prop": "categories",
                "titles": "|".join(chunk),
                "cllimit": "500",
                "clshow": "!hidden",
                "redirects": "1",
            },
        )
        for p in data.get("query", {}).get("pages", []):
            if p.get("missing"):
                out[p.get("title", "")] = []
                continue
            cats = [c["title"] for c in p.get("categories", [])]
            out[p.get("title", "")] = cats
        # redirect 별칭 보정: 요청 제목으로도 조회되게 한다
        for rd in data.get("query", {}).get("redirects", []):
            if rd.get("to") in out:
                out[rd.get("from", "")] = out[rd["to"]]
    # redirect로 제목이 정규화된 경우 요청 제목 매핑 보정
    vals = list(out.values())
    for t in chunk:
        if t not in out and vals:
            # 단일 요청 정규화 대비: 값 목록에서 순서대로 보정하지 않고
            # 호출자가 .get 후 fallback 하므로 여기서는 빈값만 보장
            out.setdefault(t, [])
    return out


def cats_for(cat_map: dict, title: str) -> list:
    if title in cat_map:
        return cat_map[title]
    vals = list(cat_map.values())
    return vals[0] if len(vals) == 1 else []


def get_links(endpoint: str, title: str) -> list:
    data = api_query(
        endpoint,
        {
            "action": "query",
            "prop": "links",
            "titles": title,
            "plnamespace": "0",
            "pllimit": "500",
            "redirects": "1",
        },
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return []
    return [lk["title"] for lk in pages[0].get("links", [])]


def get_extract(endpoint: str, title: str) -> tuple[str, str, bool]:
    """(정규 제목, 본문, missing 여부). extracts는 한 번에 한 문서만."""
    data = api_query(
        endpoint,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": title,
            "redirects": "1",
        },
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return title, "", True
    p = pages[0]
    if p.get("missing"):
        return title, "", True
    return p.get("title", title), (p.get("extract") or ""), False


def short_category(cat: str) -> str:
    for p in ("분류:", "Category:"):
        if cat.startswith(p):
            return cat[len(p):]
    return cat


def sanitize_filename(title: str) -> str:
    name = title.replace(" ", "_").replace("/", "_")
    name = re.sub(r'[\\:*?"<>|]', "_", name)
    return name.strip() or "untitled"


def save_doc(title: str, lang: str, categories: list, body: str) -> str:
    fname = f"{lang}_{sanitize_filename(title)}.md"
    path = DOCS_DIR / fname
    short_cats = [short_category(c) for c in categories]
    content = f"# {title}\n언어: {lang}\n분류: {', '.join(short_cats)}\n\n{body.strip()}\n"
    path.write_text(content, encoding="utf-8")
    return fname


def collect_lang(lang: str, endpoint: str, seeds: list, seed_cat_map: dict,
                 already_saved_titles: set, saved: list, rejected: list,
                 max_new: int) -> int:
    """해당 언어에서 시드+2홉 후보를 수집. 새로 저장한 건수 반환."""
    seed_cat_union: set = set()
    for v in seed_cat_map.values():
        seed_cat_union.update(topical(v))

    added = 0
    rejected_titles: set = set()

    def log_reject(title: str, reason: str) -> None:
        if title not in rejected_titles and title not in already_saved_titles:
            rejected_titles.add(title)
            rejected.append({"title": title, "lang": lang, "reason": reason})

    def try_save(title: str, reason: str) -> bool:
        nonlocal added
        if title in already_saved_titles or title in rejected_titles:
            return False
        if is_excluded_title(title):
            log_reject(title, "연도-목록-틀 문서")
            return False
        cats = topical(cats_for(get_categories(endpoint, [title]), title))
        shared = sorted(seed_cat_union.intersection(cats))
        if not shared:
            log_reject(title, "분류 미공유")
            return False
        real_title, body, missing = get_extract(endpoint, title)
        if missing:
            log_reject(title, "기타(문서 없음)")
            return False
        if real_title in already_saved_titles:
            return False
        if is_excluded_title(real_title):
            log_reject(real_title, "연도-목록-틀 문서")
            return False
        text = (body or "").strip()
        if len(text) < MIN_CHARS:
            log_reject(real_title, f"토막글 ({len(text)}자)")
            return False
        save_doc(real_title, lang, cats, text)
        already_saved_titles.add(title)
        already_saved_titles.add(real_title)
        saved.append(
            {
                "title": real_title,
                "lang": lang,
                "chars": len(text),
                "categories": [short_category(c) for c in cats],
                "shared": [short_category(c) for c in shared],
                "reason": reason,
            }
        )
        added += 1
        print(f"  [저장 {added}] ({lang}) {real_title} {len(text)}자 — {reason}")
        return True

    # 1) 시드 먼저
    print(f"[{lang}] 시드 {len(seeds)}건 처리 시작")
    for s in seeds:
        if added >= max_new:
            break
        # 시드는 분류 공유 조건 면제, 토막/제외 조건만 적용
        if s in already_saved_titles:
            continue
        if is_excluded_title(s):
            log_reject(s, "연도-목록-틀 문서")
            continue
        real_title, body, missing = get_extract(endpoint, s)
        if missing:
            log_reject(s, "기타(문서 없음)")
            print(f"  [탈락] ({lang}) {s} — 문서 없음")
            continue
        text = (body or "").strip()
        if len(text) < MIN_CHARS:
            log_reject(real_title, f"토막글 ({len(text)}자)")
            print(f"  [탈락] ({lang}) {real_title} — 토막글 ({len(text)}자)")
            continue
        cats = topical(cats_for(seed_cat_map, s))
        if not cats:
            cats = topical(cats_for(get_categories(endpoint, [s]), s))
        save_doc(real_title, lang, cats, text)
        already_saved_titles.add(s)
        already_saved_titles.add(real_title)
        saved.append(
            {
                "title": real_title,
                "lang": lang,
                "chars": len(text),
                "categories": [short_category(c) for c in cats],
                "reason": "시드",
            }
        )
        added += 1
        print(f"  [저장 {added}] ({lang}) {real_title} {len(text)}자 — 시드")

    # 2) 2홉 후보: 시드 본문 링크 중 여러 시드가 함께 가리키는 순
    print(f"[{lang}] 2홉 링크 수집 시작 (시드 {len(seeds)}건)")
    counter: Counter = Counter()
    for i, s in enumerate(seeds, 1):
        links = get_links(endpoint, s)
        print(f"  시드 {i}/{len(seeds)} '{s}': 링크 {len(links)}건")
        for lk in links:
            counter[lk] += 1
    # 시드 자기 자신·이미 저장분 제외, 제외 패턴 제외
    seed_set = set(seeds) | already_saved_titles
    ranked = [
        (t, c) for t, c in counter.most_common() if t not in seed_set and not is_excluded_title(t)
    ]
    # 제외 패턴에 걸린 것은 탈락 로그에 기록
    for t in counter:
        if t not in seed_set and is_excluded_title(t):
            log_reject(t, "연도-목록-틀 문서")
    print(f"[{lang}] 2홉 후보 {len(ranked)}건 (중복 제거·제외패턴 제거 후)")
    for t, c in ranked[:15]:
        print(f"    {c}개 시드 공유: {t}")

    # 분류 공유 필터 (20건씩 묶음). 호출 수 절약을 위해 풀 상한 없이 전수 검사하되
    # extracts(1건씩 호출)는 살아남은 후보에 대해서만 수행한다.
    cand_pool = [t for t, _ in ranked]
    cand_cats = get_categories(endpoint, cand_pool) if cand_pool else {}
    shared_cands = []
    for t, c in ranked:
        cats = topical(cats_for(cand_cats, t))
        if seed_cat_union.intersection(cats):
            shared_cands.append((t, c))
        else:
            log_reject(t, "분류 미공유")
    print(f"[{lang}] 분류를 공유한 후보 {len(shared_cands)}건")
    for t, c in shared_cands[:15]:
        print(f"    {c}개 시드 공유 + 분류 공유: {t}")

    for t, c in shared_cands:
        if added >= max_new:
            break
        refs = f"2홉({c}개 시드 공유+분류 공유)"
        try_save(t, refs)

    return added


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list = []
    rejected: list = []
    already: set = set()

    print(f"한국어 시드 {len(SEEDS_KO)}건, 영문 시드 {len(SEEDS_EN)}건")

    # 시드 분류 미리 확보 (20건씩 묶음)
    print("[ko] 시드 분류 조회")
    ko_seed_cats = get_categories(API_KO, SEEDS_KO)
    print("[en] 시드 분류 조회")
    en_seed_cats = get_categories(API_EN, SEEDS_EN)

    # 한국어 먼저 (최대 45건까지 시도), 이후 총합 70 목표·80 상한으로 영문 보충
    ko_added = collect_lang("ko", API_KO, SEEDS_KO, ko_seed_cats, already, saved, rejected,
                            max_new=45)
    ko_count = sum(1 for s in saved if s["lang"] == "ko")
    total = len(saved)
    print(f"[중간] ko={ko_count}, total={total}")

    need_total = max(TARGET_TOTAL_MIN - total, 0)
    need_ko = max(TARGET_KO_MIN - ko_count, 0)
    print(f"부족분: 한국어 {need_ko}건, 총합 {need_total}건 → 영문으로 보충")

    remaining = TARGET_TOTAL_AIM - total
    en_cap = max(min(remaining, TARGET_TOTAL_MAX - total), need_total)
    en_cap = max(en_cap, 0)
    if en_cap <= 0:
        en_cap = 30  # 이미 목표 달성 시에도 영문 보충분 확보
    en_added = collect_lang("en", API_EN, SEEDS_EN, en_seed_cats, already, saved,
                            rejected, max_new=en_cap)

    ko_count = sum(1 for s in saved if s["lang"] == "ko")
    en_count = sum(1 for s in saved if s["lang"] == "en")
    total = len(saved)
    print(f"[최종] ko={ko_count}, en={en_count}, total={total}")
    print("저장한 문서 목록:")
    for s in saved:
        print(f"  - ({s['lang']}) {s['title']} ({s['chars']}자)")

    manifest = {
        "seeds": {"ko": SEEDS_KO, "en": SEEDS_EN},
        "saved": saved,
        "rejected": rejected,
        "counts": {"ko": ko_count, "en": en_count, "total": total},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest 저장: {MANIFEST_PATH} (저장 {len(saved)}건, 탈락 {len(rejected)}건)")


if __name__ == "__main__":
    main()
