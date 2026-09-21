#!/usr/bin/env python3
"""W4: 디자인사 GraphRAG 평가 — 홉수별 채점 + 경로 재현율/정밀도 + basic RAG 대조 + 실패 3층 분류.

산출물: output/eval.json, output/eval_report.md
입력 읽기만 한다. data/goldenset.json · agent.py · output/graph.graphml 은 절대 수정하지 않는다.

[채점 정책 GRADING_POLICY — 엄격 기준]
- 일반 문항: 기대 정답의 핵심 개체가 답변 문자열에 들어 있어야 정답.
  핵심 개체가 일부만 있거나(예: 2개 중 1개) 관련 개체만 있으면 부분정답, 없으면 오답.
- 거절 문항(R-01/R-02): refused=True 이면 정답, 답을 내놓았으면(지어냄) 오답. 부분정답 없음.
- LLM 심판은 쓰지 않는다. 이유: (a) 결정적·재현 가능한 키워드 판정이 더 엄격하고
  (b) 심판 LLM이 피평가자와 같은 모델이면 관대해지는 편향이 생긴다.
  대신 문항별 accept 리스트와 판정 근거(reason)를 eval.json 에 남긴다.
- 점수를 잘 보이게 하려고 기준을 완화하지 않는다.

[경로 재현율/정밀도]
- expected_path 문자열을 원자 삼중항(atomic triple)으로 분할한다.
  예: "A <-TAUGHT_AT- B -FOUNDED-> C" 는 2개 원자 삼중항이다.
  따라서 H3-01/H3-03 의 분모는 4, H3-02 는 3이다. (문항의 hops 표기와 무관하게 실제 간선 수)
- 재현율 = 기대 원자 삼중항 중 실제 근거(evidence_triples)에 들어온 비율.
- 정밀도 = 실제 근거 삼중항 중 기대 경로에 속한 비율.
  W3 실행이 12문항 전부 근거 40개(상한 꽉 참)라서 재현율만 보면 과잉 탐색이 점수를
  부풀린다. 정밀도를 반드시 함께 보고한다.
- 끝점 매칭은 data/aliases.json(한영 별칭) + 코드 내 HINT_ALIASES 로 한영 표기차를 흡수한다.
  타입(Type:...) 접두사는 반드시 일치해야 한다.

[실패 3층 분류 — 자동 휴리스틱 1차 분류 + 근거 병기]
- 색인(index): 기대 원자 삼중항 중 그래프 전체에 존재하지 않는 것이 1개라도 있으면 색인.
- 탐색(retrieval): 그래프에는 있는데 근거(evidence_triples)에 없으면 탐색.
- 생성(generation): 근거에 다 있는데 답이 틀렸으면 생성.
- 거절 문항을 지어낸 경우: 그래프에 뒷받침 삼중항이 없는데 답했으면 생성(환각)으로 귀속.
"""

import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = ROOT / "data" / "goldenset.json"
RUNS_PATH = ROOT / "output" / "runs.jsonl"
GRAPH_PATH = ROOT / "output" / "graph.graphml"
ALIASES_PATH = ROOT / "data" / "aliases.json"
DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "output" / "eval.json"
REPORT_PATH = ROOT / "output" / "eval_report.md"

MODEL = "gemini-3.1-flash-lite"
BASELINE_TOP_K = 5

# --------------------------------------------------------------------------
# 채점 정책 (eval.json 에 그대로 저장됨)
# --------------------------------------------------------------------------
GRADING_POLICY = {
    "judge": "rule-based keyword match (no LLM judge — stricter and reproducible)",
    "verdicts": ["correct", "partial", "wrong"],
    "normal_items": (
        "정답(required 핵심 개체)이 모두 답변에 포함되면 correct, "
        "일부만 포함되면 partial, 없으면 wrong. "
        "required 는 기대 정답의 핵심 개체(한영 표기 모두 허용)이며, "
        "부가 설명이 붙어도 감점하지 않으나 핵심 개체가 빠지면 정답으로 치지 않는다."
    ),
    "refusal_items": (
        "refused=True 이면 correct, 답을 내놓았으면(지어냄) wrong. partial 없음. "
        "거절문에 탐색 현황이 붙는 것은 허용."
    ),
    "strictness_note": (
        "점수를 잘 보이게 기준을 완화하지 않는다. "
        "H2-04 처럼 복수 정답이 허용된 문항도 기대 집합(expected_any 중 1개 이상)이 "
        "들어가야 correct 다."
    ),
}

# qid -> {"required": [...], "expected_any": [...]}
# required: 전부 포함되어야 correct. expected_any: 1개 이상 포함되어야 correct.
# (둘 다 비어 있으면 required 만 사용. 매칭은 norm_key 부분일치, 한영 모두 기재)
ACCEPT = {
    "H1-01": {"required": ["그로피우스", "gropius"], "expected_any": []},
    "H1-02": {"required": ["브로이어", "breuer"], "expected_any": []},
    "H2-01": {"required": ["뉴바우하우스", "newbauhaus", "모호이너지", "moholy"],
              "expected_any": []},  # 학교명+설립자 실마리 중 하나라도? -> 아래 로직 참조
    "H2-02": {"required": ["하버드", "harvard"], "expected_any": []},
    "H2-03": {"required": ["바우하우스", "bauhaus"], "expected_any": []},
    "H2-04": {"required": [], "expected_any": ["몬드리안", "mondrian", "두스뷔르흐",
              "두스부르크", "doesburg"]},
    "H2-05": {"required": ["일리노이", "illinois", "iit"], "expected_any": []},
    "H3-01": {"required": ["뉴바우하우스", "newbauhaus"], "expected_any": []},
    "H3-02": {"required": [], "expected_any": ["바실리", "wassily", "체스카", "cesca"]},
    "H3-03": {"required": [], "expected_any": ["바실리", "wassily", "체스카", "cesca"]},
    "R-01": {"required": [], "expected_any": []},
    "R-02": {"required": [], "expected_any": []},
}
# H2-01 특별 규칙: 묻는 것은 '학교'이므로 학교명(뉴 바우하우스)이 필수.
# 설립자명만 있고 학교명이 없으면 partial. 이를 위해 required 를 학교명으로 좁힌다.
ACCEPT["H2-01"] = {"required": ["뉴바우하우스", "newbauhaus"], "expected_any": []}

# 베이스라인 실패 분류용: 답을 내는 데 필요한 핵심 실마리 개체(qid별).
# 검색 청크에 이 키가 없으면 검색(retrieval) 실패, 있으면 생성 실패로 귀속한다.
BRIDGE_KEYS = {
    "H1-01": ["그로피우스", "gropius"],
    "H1-02": ["브로이어", "breuer"],
    "H2-01": ["모호이너지", "모홀리", "moholy"],
    "H2-02": ["하버드", "harvard"],
    "H2-03": ["브로이어", "breuer"],
    "H2-04": ["리트벨트", "rietveld"],
    "H2-05": ["일리노이", "illinois", "iit"],
    "H3-01": ["브로이어", "breuer"],
    "H3-02": ["브로이어", "breuer"],
    "H3-03": ["브로이어", "breuer"],
}

REFUSAL_PHRASE = "근거를 찾지 못했습니다"


# --------------------------------------------------------------------------
# 정규화 / 별칭
# --------------------------------------------------------------------------
def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    for a, b in [("œ", "oe"), ("æ", "ae"), ("ø", "o"), ("ł", "l"),
                 ("đ", "d"), ("ß", "ss"), ("ı", "i")]:
        s = s.replace(a, b)
    s = re.sub(r"[\s·・∙•\-–—―_.,'\"`´‘’“”/()\[\]{}:;!?+&|]+", "", s)
    return s


def load_alias_entries():
    raw = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    entries = []
    for e in raw.get("manual", []):
        keys = {norm_key(v) for v in [e["canonical"]] + e.get("variants", [])}
        keys.discard("")
        entries.append({"canonical": e["canonical"], "type": e["type"], "keys": keys})
    return entries


ALIAS_ENTRIES = load_alias_entries()

# 별칭 사전에 없는 힌트(주로 약어·복합 표기)에 대한 명시적 매핑. 골든셋 고정 문구용.
HINT_ALIASES = {
    "iit": ["illinoisinstituteoftechnology"],
    "하버드gsd": ["harvard"],
    "하버드": ["harvard"],
    "데스테일": ["destijl"],
    "데스틸": ["destijl"],
    "신조형주의": ["destijl"],
    "바우하우스": ["bauhaus"],
    "뉴바우하우스": ["newbauhaus"],
    "바실리체어": ["wassilychair"],
    "두스뷔르흐": ["doesburg"],
    "두스부르크": ["doesburg"],
    "미스": ["miesvanderrohe"],
    "모호이너지": ["moholy"],
}


def hint_keys(hint: str):
    """힌트 문자열이 지칭할 수 있는 norm_key 집합."""
    hk = norm_key(hint)
    out = {hk}
    for k, vals in HINT_ALIASES.items():
        if k in hk or hk in k:
            out.update(vals)
    for e in ALIAS_ENTRIES:
        if any(hk in v or v in hk for v in e["keys"]):
            out.update(e["keys"])
    return out


def endpoint_match(node_id: str, node_label: str, want_type: str, hint: Optional[str]) -> bool:
    """그래프 노드가 기대 끝점 (타입 + 힌트) 에 부합하는가."""
    ntype = node_id.split(":", 1)[0] if ":" in node_id else ""
    if want_type and ntype != want_type:
        return False
    if not hint:
        return True
    nl, ni = norm_key(node_label), norm_key(node_id.split(":", 1)[-1])
    hks = hint_keys(hint)
    return any(h == nl or h == ni or h in nl or h in ni or nl in h or ni in h
               for h in hks)


# --------------------------------------------------------------------------
# expected_path 파싱
# --------------------------------------------------------------------------
REL_TOKEN = re.compile(r"(<-([A-Z_]+)-|-([A-Z_]+)->)")


def parse_expected_path(path_list):
    """expected_path 문자열 리스트 -> 원자 삼중항 리스트.

    [{left_type, left_hint, rel, arrow, right_type, right_hint}]
    arrow '->' : left -rel-> right / '<-' : right -rel-> left (정준 방향)
    """
    atomics = []
    for entry in path_list:
        parts, rels = [], []
        pos = 0
        for m in REL_TOKEN.finditer(entry):
            parts.append(entry[pos:m.start()].strip())
            rel = m.group(2) or m.group(3)
            arrow = "<-" if m.group(0).startswith("<-") else "->"
            rels.append((rel, arrow))
            pos = m.end()
        parts.append(entry[pos:].strip())
        for i, (rel, arrow) in enumerate(rels):
            l, r = parts[i], parts[i + 1]
            atomics.append({
                "left": _parse_endpoint(l), "rel": rel, "arrow": arrow,
                "right": _parse_endpoint(r), "source_entry": entry,
            })
    return atomics


def _parse_endpoint(s: str):
    m = re.match(r"^([^()]+?)(?:\((.+)\))?$", s.strip())
    typ = (m.group(1) or "").strip()
    hint = (m.group(2) or "").strip() or None
    return {"type": typ, "hint": hint}


def triple_matches(s_id, s_label, rel, o_id, o_label, atomic) -> bool:
    """정준 삼중항 (s -rel-> o) 이 기대 원자 삼중항에 부합하는가."""
    if rel != atomic["rel"]:
        return False
    L, R = atomic["left"], atomic["right"]
    if atomic["arrow"] == "->":
        return (endpoint_match(s_id, s_label, L["type"], L["hint"])
                and endpoint_match(o_id, o_label, R["type"], R["hint"]))
    else:  # '<-' : 기대 왼쪽이 목적어 쪽
        return (endpoint_match(o_id, o_label, L["type"], L["hint"])
                and endpoint_match(s_id, s_label, R["type"], R["hint"]))


def fmt_atomic(a) -> str:
    L, R = a["left"], a["right"]
    def ep(e):
        return e["type"] + (f"({e['hint']})" if e["hint"] else "")
    arr = "-{}->".format(a["rel"]) if a["arrow"] == "->" else "<-{}-".format(a["rel"])
    return f"{ep(L)} {arr} {ep(R)}"


# --------------------------------------------------------------------------
# 채점
# --------------------------------------------------------------------------
def grade_answer(qid, answer_text, refused, is_refusal) -> Tuple[str, str]:
    if is_refusal:
        if refused:
            return "correct", "거절 문항에서 거절함(refused=True). 정답."
        return "wrong", "거절 문항에서 답을 지어냄(refused=False). 오답."
    spec = ACCEPT[qid]
    hay = norm_key(answer_text or "")
    if spec["required"]:
        hits = [k for k in spec["required"] if norm_key(k) in hay]
        if hits:
            return "correct", f"핵심 개체 포함({', '.join(hits)}). 정답."
        # 부분: required 는 없지만 expected_any(있으면) 또는 질문 관련 개체가 있으면
        return "wrong", "핵심 개체가 답변에 없음. 오답."
    any_hits = [k for k in spec["expected_any"] if norm_key(k) in hay]
    if any_hits:
        return "correct", f"허용 정답 집합 포함({', '.join(any_hits)}). 정답."
    return "wrong", "허용 정답 집합 중 어느 것도 답변에 없음. 오답."


# --------------------------------------------------------------------------
# 그래프 적재
# --------------------------------------------------------------------------
def load_graph():
    import networkx as nx
    g = nx.read_graphml(str(GRAPH_PATH))
    edges = []
    for u, v, d in g.edges(data=True):
        edges.append({
            "subject_id": u, "subject": g.nodes[u].get("label") or u,
            "relation": d.get("relation"),
            "object_id": v, "object": g.nodes[v].get("label") or v,
            "source_doc": d.get("source_doc"),
            "source_sentence": d.get("source_sentence"),
        })
    return g, edges


# --------------------------------------------------------------------------
# basic RAG 베이스라인 (그래프 미사용 · BM25 top-k + 동일 LLM/유사 프롬프트)
# --------------------------------------------------------------------------
def build_chunks():
    chunks = []
    for fp in sorted(DOCS_DIR.glob("*.md")):
        text = fp.read_text(encoding="utf-8")
        #见长标题·wang단락 기준 분할 후 ~700자 합치기
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 1 <= 700:
                buf = (buf + "\n" + p).strip()
            else:
                if buf:
                    chunks.append({"doc": fp.name, "text": buf})
                buf = p if len(p) <= 1400 else p[:1400]
        if buf:
            chunks.append({"doc": fp.name, "text": buf})
    return chunks


def tokenize(s: str):
    return re.findall(r"[a-z0-9]+|[가-힣]+", s.lower())


class BM25Index:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, k=BASELINE_TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"doc": self.chunks[i]["doc"], "text": self.chunks[i]["text"],
                 "score": float(scores[i])} for i in top]


BASELINE_SYS = (
    "너는 디자인사 질의응답기다. 아래에 제시된 텍스트 근거(원문 문서 발췌)와 그 내용만으로 답하라. "
    "너의 사전 지식을 쓰지 마라. 그래프·삼중항 같은 구조화 정보는 주어지지 않는다."
)


def baseline_answer(question, retrieved) -> dict:
    from google import genai
    from google.genai import types
    key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_GENERATIVE_AI_API_KEY is not set")
    lines = [f"[{i}] (doc: {c['doc']}) {c['text'][:600]}" for i, c in enumerate(retrieved)]
    prompt = (
        f"{BASELINE_SYS}\n\n질문: {question}\n\n"
        "[텍스트 근거]\n" + ("\n\n".join(lines) if lines else "(없음)")
        + "\n\n판정 규칙:\n"
          "1. 질문이 묻는 대상이 근거 텍스트에서 직접 확인돼야 한다. 없으면 answerable=false.\n"
          "2. 질문이 전제하는 관계(예: 'X가 Y에서 가르쳤다')가 근거에 없으면 answerable=false. "
          "다른 학교·다른 작품을 끌어와 답하지 마라.\n"
          "3. 증거만으로 답을 특정할 수 없으면 answerable=false.\n"
          "출력은 JSON 한 개: "
          '{"answerable": bool, "answer": "...", "reason": "..."}. '
          "answer는 한국어로 간결히. answerable=false면 answer는 빈 문자열."
    )
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=BASELINE_SYS,
        response_mime_type="application/json",
        temperature=0.0,
    )
    last = None
    for _ in range(2):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            data = json.loads(resp.text)
            if not isinstance(data, dict):
                raise ValueError("LLM response is not a JSON object")
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        return {"answer": "", "refused": True,
                "refusal_reason": f"베이스라인 LLM 호출 실패({type(last).__name__}). 거절로 처리.",
                "llm_reason": str(last)[:200], "retrieved": retrieved}
    if not data.get("answerable"):
        return {"answer": "",
                "refused": True,
                "refusal_reason": f"{REFUSAL_PHRASE}: {data.get('reason', '텍스트 근거가 질문을 뒷받침하지 않습니다.')}",
                "llm_reason": str(data.get("reason", "")),
                "retrieved": retrieved}
    return {"answer": str(data.get("answer", "")), "refused": False,
            "refusal_reason": "", "llm_reason": str(data.get("reason", "")),
            "retrieved": retrieved}


def _same_side(t, a) -> bool:
    """근거 삼중항의 주어가 기대 원자 삼중항의 정준 주어 쪽과 같은 개체인가."""
    ep = a["left"] if a["arrow"] == "->" else a["right"]
    if not ep["hint"]:
        return False
    hks = hint_keys(ep["hint"])
    ns = norm_key(t.get("subject", ""))
    return any(h == ns or h in ns or ns in h for h in hks)


# --------------------------------------------------------------------------
# 실패 층 분류
# --------------------------------------------------------------------------
def classify_failure(verdict, atomics, in_graph_list, in_evidence_list,
                     refused, is_refusal) -> Tuple[str, str]:
    """returns (layer, reason). layer in {none, index, retrieval, generation}."""
    if verdict == "correct":
        return "none", "정답이므로 실패 층 없음."
    if is_refusal:
        # 지어냄: 그래프에 뒷받침이 없는데 답했으면 환각(생성), 있으면 탐색/색인 혼입
        if all(in_graph_list):
            return "retrieval", (
                "거절해야 할 문항인데 답을 내놓았고, 그래프에 오인될 만한 삼중항이 있다. "
                "잘못된 근거를 탐색해온 탐색층 실패로 귀속.")
        return "generation", (
            "거절해야 할 문항인데 그래프에 뒷받침 삼중항도 없이 답을 지어냄(환각). "
            "생성층 실패로 귀속.")
    missing_graph = [fmt_atomic(a) for a, f in zip(atomics, in_graph_list) if not f]
    if missing_graph:
        return "index", (
            "기대 삼중항이 그래프에 부재: " + " / ".join(missing_graph)
            + " -> 색인층(추출 실패/병합 실패).")
    missing_ev = [fmt_atomic(a) for a, f in zip(atomics, in_evidence_list) if not f]
    if missing_ev:
        return "retrieval", (
            "삼중항은 그래프에 있으나 근거에 없음: " + " / ".join(missing_ev)
            + " -> 탐색층(시작 개체 오인/상한 절단/홉 상한).")
    return "generation", "근거에 기대 삼중항이 다 있으나 답이 틀림(환각/근거 무시) -> 생성층."


# --------------------------------------------------------------------------
# 색인 결함 감사 (결정적 스캔)
# --------------------------------------------------------------------------
def audit_index_defects(g, edges):
    defects = []

    def has_edge(s_sub, rel, o_sub, doc=None):
        out = []
        for t in edges:
            if (t["relation"] == rel and s_sub in norm_key(t["subject"])
                    and o_sub in norm_key(t["object"])
                    and (doc is None or t["source_doc"] == doc)):
                out.append(t)
        return out

    # D1. 알려진 Weimar FOUNDED 오추출 (H1-01 관련, 그래프에 잔류)
    weimar_founders = has_edge("", "FOUNDED", "", doc="en_Weimar.md")
    bogus = [t for t in weimar_founders
             if norm_key(t["object"]) == "bauhaus"
             and norm_key(t["subject"]) in
             {norm_key(x) for x in ["Henry van de Velde", "Wassily Kandinsky",
                                   "Paul Klee", "Lyonel Feininger"]}]
    if bogus:
        defects.append({
            "id": "D1",
            "title": "en_Weimar.md발 FOUNDED 오추출 4건 (알려진 결함, 그래프 잔류 확인)",
            "layer": "index",
            "triples": [f"{t['subject']} -FOUNDED-> {t['object']} "
                        f"(doc: {t['source_doc']})" for t in bogus],
            "source_sentence": (bogus[0]["source_sentence"] or "")[:250],
            "why_wrong": ("원문은 'came to the city and founded the Bauhaus movement' — "
                          "사조(Movement)에 대한 서술을 학교(Institution) 설립으로 추출했고, "
                          "실제로 마이스터로 부임한 4인을 설립자로 만들었다. "
                          "Gropius의 FOUNDED는 별도 근거로 정당하므로 제외."),
            "impact": ("H1-01은 Gropius 삼중항 덕에 정답이 나오지만, "
                       "'바우하우스 설립자' 질의의 근거 정밀도를 갉아먹는 색인층 결함이다."),
        })

    # D2. 주어 오인: Grand Duke가 세운 학교를 van de Velde가 세운 것으로 추출
    d2 = [t for t in edges
          if t["relation"] == "FOUNDED"
          and "velde" in norm_key(t["subject"])
          and "grandducal" in norm_key(t["object"])]
    if d2:
        d2_nodes = sorted({t["object"] for t in d2})
        defects.append({
            "id": "D2",
            "title": "FOUNDED 주어 오인 + 동일 학교 3중복 — van de Velde 관련 Institution 노드",
            "layer": "index",
            "triples": [f"{t['subject']} -FOUNDED-> {t['object']} "
                        f"(doc: {t['source_doc']})" for t in d2],
            "source_sentence": (d2[0]["source_sentence"] or "")[:250],
            "why_wrong": ("(a) en_Bauhaus.md 삼중항은 명백한 주어 오인이다. 원문 주어는 "
                          "'the Grand Duke of Saxe-Weimar-Eisenach'이고 van de Velde는 "
                          "'directed by'의 대상(교장)이다. 수동문 주어를 설립자로 오인한 추출 오류. "
                          "(b) en_Art_Nouveau.md 삼중항('he founded the Grand-Ducal School')은 "
                          "문장 자체는 van de Velde 설립을 서술하나, 결과적으로 같은 학교가 "
                          "'Grand-Ducal School of Arts and Crafts' / "
                          "'Grand Ducal Saxon School of Arts and Crafts' / "
                          "'Kunstgewerbeschule Weimar' 3개 Institution 노드로 중복됐다. "
                          "별칭 병합 실패(색인층)다."),
            "impact": ("설립자 질의에서 오답 후보를 만들고, 동일 학교가 3개 노드로 분단돼 "
                       "TAUGHT_AT/FOUNDED 경로 탐색을 분산시킨다."),
        })

    # D3. 역사 서술 라벨을 Institution으로: Second School of Chicago
    d3 = [t for t in edges
          if t["relation"] == "FOUNDED" and "secondschoolofchicago" in norm_key(t["object"])]
    if d3:
        defects.append({
            "id": "D3",
            "title": "역사 서술 라벨의 Institution 오분류 — 'Second School of Chicago'",
            "layer": "index",
            "triples": [f"{t['subject']} -FOUNDED-> {t['object']} "
                        f"(node_type=Institution, doc: {t['source_doc']})" for t in d3],
            "source_sentence": (d3[0]["source_sentence"] or "")[:250],
            "why_wrong": ("'Second School of Chicago'는 건축사 서술상의 유파 라벨이지 "
                          "설립 가능한 학교/회사가 아니다. 스키마(FOUNDED: Person->Institution|Company) 위반."),
            "impact": "Mies 관련 설립 질의의 정밀도를 떨어뜨리는 색인층 결함.",
        })

    # D4. 예술 그룹을 Company로: 네오모던
    d4 = [t for t in edges
          if t["relation"] == "FOUNDED" and "네오모던" in t["object"]]
    if d4:
        defects.append({
            "id": "D4",
            "title": "예술 그룹의 Company 오분류 — '네오모던' (FOUNDED 대상 오용)",
            "layer": "index",
            "triples": [f"{t['subject']} -FOUNDED-> {t['object']} "
                        f"(node_type=Company, doc: {t['source_doc']})" for t in d4],
            "source_sentence": (d4[0]["source_sentence"] or "")[:250],
            "why_wrong": ("'네오모던 예술가 그룹'은 운동/집단이지 회사가 아니다. "
                          "FOUNDED의 대상은 Institution|Company여야 하므로 스키마 위반."),
            "impact": "FOUNDED 관계 전체의 신뢰도를 갉아먹는 추출 오류.",
        })

    # D5. 일반명사 Work 노드 (§4.4 위반) — 규모 측정 + 대표 예시.
    # 엄격 기준: 소문자로 시작하거나 일반명사 denylist 에 들어가는 라벨만 집계한다.
    # (대문자 고유명사형 작품명은 제외. 소문자 스타일 정식 명칭이 극소수 섞일 수 있어
    #  '후보' 수치로 보고하고 대표 예시는 명백한 일반명사로만 든다.)
    generic_deny = {
        "products", "houses", "buildings", "building", "clothing", "jewelry",
        "jewellery", "dresses", "bottles", "furniture", "chairs", "posters",
        "paintings", "sculptures", "costumes and sets", "art nouveau designs",
        "modernist furniture designs", "a floor", "a massive fireplace",
        "balustrades", "bridges and service stations", "exhibition house",
        "glass fountain", "glass pavilion", "crematorium", "alphabet",
    }
    generic_nodes = []
    for n in g.nodes:
        if g.nodes[n].get("type") != "Work":
            continue
        lab = (g.nodes[n].get("label") or "").strip()
        if lab[:1].islower() or lab.lower() in generic_deny:
            generic_nodes.append({"id": n, "label": lab, "degree": int(g.degree(n))})
    if generic_nodes:
        generic_nodes.sort(key=lambda d: (-d["degree"], d["label"]))
        clear_first = sorted(
            generic_nodes,
            key=lambda d: (0 if d["label"].lower() in generic_deny else 1,
                           -d["degree"], d["label"]))
        defects.append({
            "id": "D5",
            "title": f"일반명사 Work 노드 약 {len(generic_nodes)}개 잔류 후보 (GUIDE §4.4 위반)",
            "layer": "index",
            "count": len(generic_nodes),
            "examples": [f"{d['label']} (deg {d['degree']})" for d in clear_first[:15]],
            "why_wrong": ("'building', 'houses', 'clothing', 'jewelry', 'products', "
                          "'art nouveau designs', 'costumes and sets' 등 일반명사가 "
                          "Work 노드로 추출됐다. §4에서 제외하기로 한 것들이다. "
                          "집계는 소문자 시작·일반명사형만 센 엄격 기준이며, 소문자 스타일 "
                          "정식 명칭이 극소수 섞여 있을 수 있다."),
            "impact": ("대부분 deg 1이라 이번 12문항 채점에는 직접 영향이 없으나, "
                       "DESIGNED 관계(935건)의 정밀도를 구조적으로 떨어뜨린다."),
        })

    # R-01/R-02 거절 정당성 확인 (색인층이 아닌, 그래프에 뒷받침이 정말 없는지)
    r01_support = [t for t in edges
                   if t["relation"] == "DESIGNED" and "breuer" in norm_key(t["subject"])
                   and any(k in norm_key(t["object"]) for k in ["car", "auto", "vehicle", "자동차"])]
    rams_bauhaus = [t for t in edges
                    if t["relation"] == "TAUGHT_AT" and "rams" in norm_key(t["subject"])
                    and "bauhaus" in norm_key(t["object"])]
    defects.append({
        "id": "CHECK-R",
        "title": "거절 문항의 그래프 내 뒷받침 존재 여부 (오답이 아닌지 확인용)",
        "layer": "index",
        "breuer_car_designed_triples": len(r01_support),
        "rams_taught_at_bauhaus_triples": len(rams_bauhaus),
        "why_wrong": "해당 없음(확인용 항목).",
        "impact": ("두 값 모두 0이어야 거절이 정당하다. 0이 아니면 거절 문항 설계 자체가 깨진다. "
                   "실측: Breuer-자동차 0건, Rams-TAUGHT_AT-Bauhaus 0건 -> 거절 정당."),
    })
    return defects


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------
def load_runs():
    rows = [json.loads(l) for l in RUNS_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return {r["qid"]: r for r in rows}


def main():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["items"]
    runs = load_runs()
    print(f"golden={len(golden)} runs={len(runs)}")
    g, edges = load_graph()
    print(f"graph nodes={g.number_of_nodes()} edges={len(edges)}")

    # 기대 원자 삼중항의 그래프 존재 여부 (실패 분류용, 문항 공통)
    atomics_by_qid = {it["id"]: parse_expected_path(it.get("expected_path", []))
                      for it in golden}
    in_graph_by_qid = {}
    for qid, atomics in atomics_by_qid.items():
        flags = []
        for a in atomics:
            hit = any(triple_matches(t["subject_id"], t["subject"], t["relation"],
                                     t["object_id"], t["object"], a) for t in edges)
            flags.append(hit)
        in_graph_by_qid[qid] = flags

    # basic RAG 베이스라인 (그래프 미사용). 결과 캐시로 LLM 재호출을 피한다.
    chunks = build_chunks()
    print(f"baseline chunks={len(chunks)} from {len(list(DOCS_DIR.glob('*.md')))} docs")
    index = BM25Index(chunks)
    cache_path = ROOT / "output" / "baseline_cache.json"
    if cache_path.exists():
        baseline_by_qid = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"baseline cache hit: {len(baseline_by_qid)} items (no LLM calls)")
    else:
        baseline_by_qid = {}
        for it in golden:
            qid = it["id"]
            ret = index.search(it["question"], k=BASELINE_TOP_K)
            b = baseline_answer(it["question"], ret)
            baseline_by_qid[qid] = b
            print(f"[baseline {qid}] refused={b['refused']} ans={(b['answer'] or '')[:60]}",
                  flush=True)
        cache_path.write_text(json.dumps(baseline_by_qid, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"wrote {cache_path}")

    # 문항별 평가
    items = []
    for it in golden:
        qid = it["id"]
        is_ref = it.get("type") == "refusal" or it.get("hops") is None
        bucket = "refusal" if is_ref else f"{it['hops']}hop"
        atomics = atomics_by_qid[qid]
        in_graph = in_graph_by_qid[qid]
        r = runs[qid]

        ev = r.get("evidence_triples", [])
        in_ev, matched_ev_idx = [], []
        for a in atomics:
            hit_idx = [i for i, t in enumerate(ev)
                       if triple_matches(t.get("subject_id", ""), t.get("subject", ""),
                                         t.get("relation", ""), t.get("object_id", ""),
                                         t.get("object", ""), a)]
            in_ev.append(bool(hit_idx))
            matched_ev_idx.extend(hit_idx)
        matched_ev_idx = sorted(set(matched_ev_idx))
        n_ev = len(ev)
        recall = (sum(in_ev) / len(atomics)) if atomics else None
        precision = (len(matched_ev_idx) / n_ev) if (atomics and n_ev) else None

        g_verdict, g_reason = grade_answer(qid, r.get("answer", ""),
                                           r.get("refused", False), is_ref)
        g_layer, g_layer_reason = classify_failure(
            g_verdict, atomics, in_graph, in_ev, r.get("refused", False), is_ref)

        b = baseline_by_qid[qid]
        b_verdict, b_reason = grade_answer(qid, b.get("answer", ""),
                                           b.get("refused", False), is_ref)
        # 베이스라인 실패 분류: 그래프 존재 여부 + 검색 청크 내 핵심어 존재로 근사
        if b_verdict == "correct":
            b_layer, b_layer_reason = "none", "정답이므로 실패 층 없음."
        else:
            missing_graph = [fmt_atomic(a) for a, f in zip(atomics, in_graph) if not f]
            if is_ref:
                b_layer = "generation"
                b_layer_reason = ("거절해야 할 문항인데 텍스트만으로 답을 지어냄(환각). "
                                  "basic RAG 생성층 실패로 귀속.")
            elif missing_graph:
                b_layer = "index"
                b_layer_reason = ("기대 삼중항이 그래프는커녕 원문 근거로도 성립하지 않을 가능성: "
                                  + " / ".join(missing_graph))
            else:
                hay = norm_key(" ".join(c["text"] for c in b["retrieved"]))
                bkeys = BRIDGE_KEYS.get(qid, [])
                if bkeys and not any(norm_key(k) in hay for k in bkeys):
                    b_layer = "retrieval"
                    b_layer_reason = ("검색된 top-k 청크에 답의 핵심 실마리 "
                                      f"({', '.join(bkeys)})가 없음 "
                                      "-> basic RAG 검색(retrieval) 실패로 귀속.")
                else:
                    b_layer = "generation"
                    b_layer_reason = ("청크에 실마리가 있으나 답을 못 냄 "
                                      "-> basic RAG 생성 실패로 귀속.")

        # grounding 미세 결함: 정답이지만 기대 원자 삼중항 일부가 근거에 없음.
        # 형제 간선(같은 주어+관계, 다른 목적어) 또는 근거 문장 내 언급이
        # 답을 우회 뒷받침하는지 함께 기록한다.
        grounding_note = ""
        if g_verdict == "correct" and atomics and not all(in_ev):
            missing = [a for a, f in zip(atomics, in_ev) if not f]
            sib_lines = []
            for a in missing:
                sibs = []
                textual = []
                ep_subj = a["left"] if a["arrow"] == "->" else a["right"]
                ep_obj = a["right"] if a["arrow"] == "->" else a["left"]
                for t in ev:
                    if (t.get("relation") == a["rel"]
                            and _same_side(t, a)
                            and not triple_matches(
                                t.get("subject_id", ""), t.get("subject", ""),
                                t.get("relation", ""), t.get("object_id", ""),
                                t.get("object", ""), a)):
                        sibs.append(f"{t['subject']} -{t['relation']}-> {t['object']}")
                    sent = norm_key(t.get("source_sentence") or "")
                    if sent and ep_subj["hint"] and ep_obj["hint"]:
                        sk = hint_keys(ep_subj["hint"])
                        ok = hint_keys(ep_obj["hint"])
                        if (any(k in sent for k in sk if len(k) > 3)
                                and any(k in sent for k in ok if len(k) > 3)
                                and f"{t['subject']} -{t['relation']}-> {t['object']}"
                                not in sibs):
                            textual.append(
                                f"문장 언급({t['source_doc']}): "
                                f"{(t.get('source_sentence') or '')[:90]}…")
                support = " / ".join(sibs[:3] + textual[:2])
                sib_lines.append(fmt_atomic(a) + " | 우회 근거: "
                                 + (support if support else "없음(파라메트릭 지식 의심)"))
            grounding_note = ("정답이나 기대 경로 일부가 근거 40개에 없음(상한 절단). "
                              "답은 아래 우회 근거/원문 문장으로 도출됨: " + " // ".join(sib_lines))

        items.append({
            "qid": qid, "bucket": bucket, "hops": it.get("hops"),
            "question": it["question"],
            "expected_answer": it["expected_answer"],
            "expected_path": it.get("expected_path", []),
            "expected_atomics": [fmt_atomic(a) for a in atomics],
            "expected_atomics_in_graph": in_graph,
            "graphrag": {
                "answer": r.get("answer", ""), "refused": r.get("refused", False),
                "verdict": g_verdict, "verdict_reason": g_reason,
                "hops_taken": r.get("hops_taken", 0),
                "n_evidence": n_ev,
                "path_recall": recall,
                "expected_atomics_in_evidence": in_ev,
                "path_precision": precision,
                "n_evidence_on_path": len(matched_ev_idx),
                "failure_layer": g_layer,
                "failure_reason": g_layer_reason,
                "grounding_note": grounding_note,
                "truncated_by_cap": r.get("truncated_by_cap", False),
                "fanout_truncations": r.get("fanout_truncations", 0),
                "evidence_truncated": r.get("evidence_truncated", 0),
            },
            "baseline": {
                "answer": b.get("answer", ""), "refused": b.get("refused", False),
                "verdict": b_verdict, "verdict_reason": b_reason,
                "failure_layer": b_layer, "failure_reason": b_layer_reason,
                "llm_reason": b.get("llm_reason", ""),
                "retrieved": [{"doc": c["doc"], "score": round(c["score"], 3),
                               "text": c["text"][:300]} for c in b["retrieved"]],
            },
        })

    # 홉수별 집계 — 전체 평균 없음(실격 조항 회피: average 필드 자체를 두지 않음)
    by_hop = {}
    for bucket in ["1hop", "2hop", "3hop", "refusal"]:
        sub = [x for x in items if x["bucket"] == bucket]
        if not sub:
            continue
        g_ok = sum(1 for x in sub if x["graphrag"]["verdict"] == "correct")
        g_part = sum(1 for x in sub if x["graphrag"]["verdict"] == "partial")
        b_ok = sum(1 for x in sub if x["baseline"]["verdict"] == "correct")
        b_part = sum(1 for x in sub if x["baseline"]["verdict"] == "partial")
        recs = [x["graphrag"]["path_recall"] for x in sub
                if x["graphrag"]["path_recall"] is not None]
        precs = [x["graphrag"]["path_precision"] for x in sub
                 if x["graphrag"]["path_precision"] is not None]
        by_hop[bucket] = {
            "n": len(sub),
            "qids": [x["qid"] for x in sub],
            "graphrag": {
                "correct": g_ok, "partial": g_part,
                "wrong": len(sub) - g_ok - g_part,
                "accuracy": round(g_ok / len(sub), 3),
            },
            "baseline": {
                "correct": b_ok, "partial": b_part,
                "wrong": len(sub) - b_ok - b_part,
                "accuracy": round(b_ok / len(sub), 3),
            },
            "accuracy_gap_graphrag_minus_baseline":
                round(g_ok / len(sub) - b_ok / len(sub), 3),
            "mean_path_recall": round(sum(recs) / len(recs), 3) if recs else None,
            "mean_path_precision": round(sum(precs) / len(precs), 3) if precs else None,
            "all_hops_taken_3_and_evidence_capped": all(
                x["graphrag"]["hops_taken"] == 3 and x["graphrag"]["n_evidence"] == 40
                for x in sub),
        }

    defects = audit_index_defects(g, edges)

    eval_doc = {
        "meta": {
            "model": MODEL,
            "baseline": (f"BM25(rank_bm25) top-{BASELINE_TOP_K} chunks, no graph, "
                         f"same LLM({MODEL}), same prompt skeleton (text evidence only)"),
            "n_items": len(items),
            "buckets": ["1hop", "2hop", "3hop", "refusal"],
            "note": ("홉수별 분리 보고만 한다. 전체 평균(accuracy 평균)은 과제 실격 조항이므로 "
                     "산출하지 않는다."),
        },
        "grading_policy": GRADING_POLICY,
        "accept_lists": ACCEPT,
        "bridge_keys": BRIDGE_KEYS,
        "items": items,
        "by_hop": by_hop,
        "index_defects": defects,
        "limitations": [
            "조기 종료 없음: 답을 찾아도 3홉까지 확장하므로 12문항 전부 hops_taken=3, "
            "근거 40개(상한)다. 정밀도 하락의 직접 원인. GUIDE §7 알려진 한계.",
            "거절 문항의 경로 재현율/정밀도는 기대 경로가 없어 측정 불가(null).",
            "기본 RAG 청크 분할(~700자)·토크나이저(영숫자/한글 블록)는 단순 구현이며, "
            "임베딩 기반 검색이 아니라는 한계가 있다.",
        ],
    }
    EVAL_PATH.write_text(json.dumps(eval_doc, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"wrote {EVAL_PATH}")

    # 사람이 읽는 보고서 (REPORT.md 에 그대로 옮길 수 있는 표 형태)
    L = []
    L.append("# W4 평가 보고서 (GraphRAG vs basic RAG, 홉수별 분리)")
    L.append("")
    L.append(f"- 모델: {MODEL} / 베이스라인: BM25 top-{BASELINE_TOP_K} 텍스트 청크, 그래프 미사용")
    L.append("- 채점: 결정적 키워드 판정(엄격). LLM 심판 미사용.")
    L.append("- 전체 평균은 내지 않는다(홉수별 분리 보고가 과제 기준).")
    L.append("")
    L.append("## 1. 홉수별 정답률 (GraphRAG vs basic RAG)")
    L.append("")
    L.append("| 버킷 | n | GraphRAG 정답 | basic RAG 정답 | 격차(G−B) |")
    L.append("|---|---|---|---|---|")
    for bucket in ["1hop", "2hop", "3hop", "refusal"]:
        s = by_hop[bucket]
        L.append(f"| {bucket} | {s['n']} | "
                 f"{s['graphrag']['correct']}/{s['n']} ({s['graphrag']['accuracy']:.0%}) | "
                 f"{s['baseline']['correct']}/{s['n']} ({s['baseline']['accuracy']:.0%}) | "
                 f"{s['accuracy_gap_graphrag_minus_baseline']:+.0%} |")
    L.append("")
    L.append("## 2. 경로 재현율 / 정밀도 (홉수별, GraphRAG)")
    L.append("")
    L.append("| 버킷 | 평균 재현율 | 평균 정밀도 | 비고 |")
    L.append("|---|---|---|---|")
    for bucket in ["1hop", "2hop", "3hop", "refusal"]:
        s = by_hop[bucket]
        rec = "측정 불가(기대 경로 없음)" if s["mean_path_recall"] is None else f"{s['mean_path_recall']:.3f}"
        prec = "측정 불가(기대 경로 없음)" if s["mean_path_precision"] is None else f"{s['mean_path_precision']:.3f}"
        note = "전부 3홉·근거 40개(상한)" if s["all_hops_taken_3_and_evidence_capped"] else ""
        L.append(f"| {bucket} | {rec} | {prec} | {note} |")
    L.append("")
    L.append("## 3. 문항별 상세 (GraphRAG)")
    L.append("")
    L.append("| qid | 판정 | 재현율 | 정밀도 | 답변 | 실패층 |")
    L.append("|---|---|---|---|---|---|")
    for x in items:
        gr = x["graphrag"]
        rec = "-" if gr["path_recall"] is None else f"{gr['path_recall']:.2f}"
        prec = "-" if gr["path_precision"] is None else f"{gr['path_precision']:.2f}"
        ans = (gr["answer"] or "").replace("|", "/").replace("\n", " ")[:60]
        L.append(f"| {x['qid']} | {gr['verdict']} | {rec} | {prec} | {ans} | {gr['failure_layer']} |")
    L.append("")
    L.append("## 4. 문항별 상세 (basic RAG)")
    L.append("")
    L.append("| qid | 판정 | 답변 | 실패층 | 실패 근거 |")
    L.append("|---|---|---|---|---|")
    for x in items:
        b = x["baseline"]
        ans = (b["answer"] or b.get("refusal_reason", "")).replace("|", "/").replace("\n", " ")[:60]
        rs = (b["failure_reason"] or "").replace("|", "/")[:80]
        L.append(f"| {x['qid']} | {b['verdict']} | {ans} | {b['failure_layer']} | {rs} |")
    L.append("")
    L.append("## 5. 실패 층 분류 (GraphRAG, 오답만 귀속)")
    L.append("")
    wrong = [x for x in items if x["graphrag"]["verdict"] != "correct"]
    if not wrong:
        L.append("- 오답 0건: 귀속할 실패 없음(현행 12문항·엄격 기준).")
    else:
        for x in wrong:
            L.append(f"- {x['qid']}: {x['graphrag']['failure_layer']} — {x['graphrag']['failure_reason']}")
    L.append("")
    L.append("## 5b. Grounding 미세 결함 (정답이나 기대 경로 일부가 근거에 없음)")
    L.append("")
    gaps = [x for x in items if x["graphrag"].get("grounding_note")]
    if not gaps:
        L.append("- 없음: 모든 정답 문항의 기대 경로가 근거에 완전히 포함됨.")
    else:
        for x in gaps:
            L.append(f"- {x['qid']}: {x['graphrag']['grounding_note']}")
    L.append("")
    L.append("## 6. 색인 결함 목록 (그래프 실측 기반)")
    L.append("")
    for d in defects:
        L.append(f"### {d['id']}. {d['title']}")
        L.append(f"- 층: {d['layer']}")
        for t in d.get("triples", [])[:6]:
            L.append(f"  - `{t}`")
        if d.get("count") is not None:
            L.append(f"  - 규모: {d['count']}개")
            for e in d.get("examples", [])[:15]:
                L.append(f"    - {e}")
        for k in ["source_sentence", "why_wrong", "impact",
                  "breuer_car_designed_triples", "rams_taught_at_bauhaus_triples"]:
            if k in d:
                L.append(f"- {k}: {d[k]}")
        L.append("")
    L.append("## 7. 막힌 지점 / 한계")
    for lim in eval_doc["limitations"]:
        L.append(f"- {lim}")
    L.append("")
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()