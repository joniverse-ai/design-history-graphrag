#!/usr/bin/env python3
"""W3: 디자인사 GraphRAG 에이전트 — 시작 개체 연결 → 최대 3홉 확장 → 근거 답변/거절.

LangGraph StateGraph 구조:
    link_entities -> expand -> collect_evidence -> decide -> generate | refuse

상태(State): question, start_entities, hops_taken, visited_paths, evidence_triples,
             truncated_by_cap, answer, refused, refusal_reason
내부 보조: mentions(언급 그룹), required_rels, movement_prunes, visited_nodes

정책 (GUIDE.md §5 v2 — 측정 후 개정):
- 홉 상한 3. 3홉에서 끊고 답이 없으면 넓히지 않고 거절.
- 허브 경유 허용. 차수 기반 경유 금지는 폐기.
- Movement 노드 경유는 경로당 1회로 제한 (유지).
- 팬아웃 상한: 노드당 최대 FANOUT_CAP(기본 12)개만 확장 (관계 타입 우선).
- 근거 상한: LLM에 넘기는 삼중항 최대 EVIDENCE_CAP(기본 40)개 (경로상 엣지 우선).
- 상한 절단은 truncated_by_cap=True 로 기록 (실패와 별도 집계).

개체 연결: data/aliases.json 역방향 인덱스 + 정규화 키(공백·중점·하이픈·대소문자·
디아크리틱 흡수)로 한글 표기 -> 정준 노드. 못 붙이면 억지 매칭 없이 거절 사유로 기록.

LLM: google-genai, gemini-3.1-flash-lite. 키는 환경변수 GOOGLE_GENERATIVE_AI_API_KEY.
수집한 근거 삼중항만으로 답을 만들며, 근거가 질문의 관계/대상을 뒷받침하지 못하면 거절.

라이브러리 진입점: answer_question(question: str) -> dict
일괄 실행: python agent.py  (data/goldenset.json 12문항 -> output/runs.jsonl)
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "output" / "graph.graphml"
ALIASES_PATH = ROOT / "data" / "aliases.json"
GOLDEN_PATH = ROOT / "data" / "goldenset.json"
RUNS_PATH = ROOT / "output" / "runs.jsonl"
DIAGRAM_PATH = ROOT / "output" / "agent_architecture.mmd"

CONFIG_PATH = ROOT / "config.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

MODEL = CFG["generation"]["model"]

_tv = CFG["traversal"]          # GUIDE.md §5 v2
MAX_HOPS = _tv["max_hops"]
MAX_MOVEMENT_TRANSIT = _tv["max_movement_transit"]
MAX_PATHS = _tv["max_paths"]
FANOUT_CAP = _tv["fanout_cap"]
EVIDENCE_CAP = _tv["evidence_cap"]

RELATIONS = set(CFG["schema"]["relations"])

# 질문 키워드 -> 요구 관계 타입 (decide 단계의 관계 적합성 검사 + LLM 힌트)
REL_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("FOUNDED", ["설립", "세운", "세웠", "창립", "창설", "만든 학교", "세운 학교"]),
    ("TAUGHT_AT", ["가르쳤", "가르친", "가르쳤던", "교수", "강의", "교장", "과목", "가르치", "지도교수", "이끈"]),
    ("STUDIED_AT", ["수학", "졸업", "배웠", "입학", "공부했", "다녔"]),
    ("DESIGNED", ["디자인한", "디자인했", "작품", "만든", "설계한", "체어"]),
    ("BELONGS_TO", ["속한", "사조", "소속", "일원", "멤버", "운동에"]),
    ("MANUFACTURED_BY", ["생산", "제조", "생산한", "만든 회사"]),
    ("INFLUENCED_BY", ["영향"]),
]


# ---------------------------------------------------------------- 정규화/별칭

def norm_label(s: str) -> str:
    """W2 build_graph.py 의 norm 과 동일: 공백 정리 + 소문자."""
    return re.sub(r"\s+", " ", s.strip()).lower()


def norm_key(s: str) -> str:
    """표기 흔들림 흡수 키: 공백·중점·하이픈·대소문자·디아크리틱 제거."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    for a, b in [("œ", "oe"), ("æ", "ae"), ("ø", "o"), ("ł", "l"),
                 ("đ", "d"), ("ß", "ss"), ("ı", "i")]:
        s = s.replace(a, b)
    s = re.sub(r"[\s·・∙•\-–—―_.,'\"`´‘’“”/()\[\]{}:;!?+&|]+", "", s)
    return s


_ALIAS_INDEX: Optional[List[Dict[str, Any]]] = None  # 긴 변형 우선 정렬


def load_alias_index() -> List[Dict[str, Any]]:
    global _ALIAS_INDEX
    if _ALIAS_INDEX is not None:
        return _ALIAS_INDEX
    raw = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    idx: List[Dict[str, Any]] = []
    for e in raw.get("manual", []):
        canon, ntype = e["canonical"], e["type"]
        seen = set()
        for v in [canon] + e.get("variants", []):
            k = norm_key(v)
            if not k or k in seen:
                continue
            seen.add(k)
            idx.append({"key": k, "canonical": canon, "type": ntype, "variant": v})
    idx.sort(key=lambda d: -len(d["key"]))  # 긴(구체적) 표기 우선
    _ALIAS_INDEX = idx
    return idx


def _load_caps() -> None:
    """config.json 이 있으면 FANOUT_CAP/EVIDENCE_CAP 을 덮어쓴다 (없으면 기본값)."""
    global FANOUT_CAP, EVIDENCE_CAP
    cfg = ROOT / "config.json"
    if not cfg.exists():
        return
    try:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        if isinstance(raw.get("FANOUT_CAP"), int) and raw["FANOUT_CAP"] > 0:
            FANOUT_CAP = raw["FANOUT_CAP"]
        if isinstance(raw.get("EVIDENCE_CAP"), int) and raw["EVIDENCE_CAP"] > 0:
            EVIDENCE_CAP = raw["EVIDENCE_CAP"]
    except Exception:
        pass


_load_caps()

# ---------------------------------------------------------------- 그래프

_G = None
_NTYPE: Optional[Dict[str, str]] = None


def load_graph():
    """GUIDE.md §5 v2: 차수 기반 허브 금지는 폐기. 허브 집합을 계산하지 않는다."""
    global _G, _NTYPE
    if _G is not None:
        return _G, set(), _NTYPE
    import networkx as nx
    _G = nx.read_graphml(str(GRAPH_PATH))
    _NTYPE = {n: (_G.nodes[n].get("type") or "") for n in _G.nodes}
    return _G, set(), _NTYPE


def node_label(nid: str) -> str:
    g, _, _ = load_graph()
    return g.nodes[nid].get("label") or nid


def incident_triples(nid: str) -> List[Dict[str, Any]]:
    """노드에 닿은 모든 엣트를 주어-관계-목적어(방향 보정) 삼중항으로.

    동일 (주어, 관계, 목적어)의 병렬 엣지는 하나로 합치되 support(추출 중복 횟수,
    여러 문서에서 반복 확인된 정도)를 남긴다. support가 낮을수록 추출 노이즈일
    가능성이 크다는 신호로 생성 단계에서 가중치로 쓴다.
    """
    g, _, _ = load_graph()
    out: List[Dict[str, Any]] = []
    seen = set()
    counts: Dict[Tuple[str, str, str], int] = {}
    order: List[Tuple[str, str, str]] = []
    first: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for nbr in list(g.predecessors(nid)) + list(g.successors(nid)):
        pairs = []
        if g.has_edge(nid, nbr):
            pairs += [(nid, nbr, d) for d in g[nid][nbr].values()] if g.is_multigraph() \
                else [(nid, nbr, g[nid][nbr])]
        if g.has_edge(nbr, nid):
            pairs += [(nbr, nid, d) for d in g[nbr][nid].values()] if g.is_multigraph() \
                else [(nbr, nid, g[nbr][nid])]
        for s, o, d in pairs:
            t = (s, d.get("relation"), o)
            counts[t] = counts.get(t, 0) + 1
            if t in seen:
                continue
            seen.add(t)
            order.append(t)
            first[t] = {
                "subject_id": s, "subject": node_label(s),
                "relation": d.get("relation"), "object_id": o, "object": node_label(o),
                "source_doc": d.get("source_doc"), "source_sentence": d.get("source_sentence"),
            }
    for t in order:
        first[t]["support"] = counts[t]
        out.append(first[t])
    return out


# ---------------------------------------------------------------- 개체 연결

def link_entities(question: str) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """질문 속 한글 표기를 정준 노드 id 로 연결.

    Returns: (start_entities, mentions, unlinked)
    mentions: [{mention, nodes:[nid...]}] — 같은 스팬에서 나온 노드들의 그룹.
    """
    g, _, _ = load_graph()
    idx = load_alias_index()
    qk = norm_key(question)
    accepted: List[Tuple[int, int]] = []  # (start, end) — 긴 매칭 우선, 겹침 제거
    matched: List[Dict[str, Any]] = []
    for e in idx:
        k = e["key"]
        if len(k) < 2:
            continue
        pos = 0
        while True:
            i = qk.find(k, pos)
            if i < 0:
                break
            span = (i, i + len(k))
            # 동일 스팬의 다른 타입 매칭(예: 바우하우스 Institution/Movement)은 유지하고,
            # 더 긴 매칭에 완전히 포함되는 짧은 매칭만 제거한다.
            if span in accepted or not any(s < span[1] and span[0] < t for s, t in accepted):
                accepted.append(span)
                matched.append({**e, "span": span})
            pos = i + 1
    # 노드 id 확정: 정준명 우선, 없으면 변형 표기로. 그래프에 없으면 unlinked.
    starts: List[str] = []
    mentions: List[Dict[str, Any]] = []
    unlinked: List[str] = []
    by_span: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for m in matched:
        by_span.setdefault(m["span"], []).append(m)
    for span in sorted(by_span):
        nodes: List[str] = []
        for m in by_span[span]:
            cands = [f"{m['type']}:{norm_label(m['canonical'])}"]
            # 폴백: 변형 표기 그대로의 노드 id (억지 유사 매칭은 하지 않음)
            for v in [m["variant"]]:
                cid = f"{m['type']}:{norm_label(v)}"
                if cid not in cands:
                    cands.append(cid)
            hit = next((c for c in cands if c in g), None)
            if hit and hit not in nodes and hit not in starts:
                nodes.append(hit)
                starts.append(hit)
            if not hit and m["canonical"] not in unlinked:
                unlinked.append(f"{m['canonical']}({m['type']})")
        if nodes:
            mentions.append({"mention_key": qk[span[0]:span[1]], "nodes": nodes})
    return starts, mentions, unlinked


def infer_relations(question: str) -> List[str]:
    return [rel for rel, kws in REL_KEYWORDS if any(k in question for k in kws)]


# ---------------------------------------------------------------- LangGraph 상태/노드

from typing_extensions import TypedDict  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402


class AgentState(TypedDict, total=False):
    question: str
    start_entities: List[str]
    mentions: List[Dict[str, Any]]
    unlinked: List[str]
    required_rels: List[str]
    hops_taken: int
    visited_paths: List[Dict[str, Any]]
    visited_nodes: List[str]
    reachable: Dict[str, List[str]]      # 그룹키 -> 도달 노드 (연결 판정용)
    evidence_triples: List[Dict[str, Any]]
    truncated_by_cap: bool
    fanout_truncations: int
    evidence_truncated: int
    movement_prunes: int
    connected: bool
    answer: str
    refused: bool
    refusal_reason: str
    llm_reason: str


def _edge_triples(cur: str, nb: str) -> List[Dict[str, Any]]:
    g, _, _ = load_graph()
    trips: List[Dict[str, Any]] = []
    seen = set()
    pairs = []
    if g.has_edge(cur, nb):
        pairs += [(cur, nb, d) for d in g[cur][nb].values()]
    if g.has_edge(nb, cur):
        pairs += [(nb, cur, d) for d in g[nb][cur].values()]
    for s, o, d in pairs:
        t = (s, d.get("relation"), o)
        if t in seen:
            continue
        seen.add(t)
        trips.append({
            "subject_id": s, "subject": node_label(s),
            "relation": d.get("relation"), "object_id": o, "object": node_label(o),
            "source_doc": d.get("source_doc"), "source_sentence": d.get("source_sentence"),
        })
    return trips


def node_link_entities(state: AgentState) -> Dict[str, Any]:
    starts, mentions, unlinked = link_entities(state["question"])
    return {"start_entities": starts, "mentions": mentions, "unlinked": unlinked,
            "required_rels": infer_relations(state["question"])}


def _ranked_neighbors(cur: str, required_rels: List[str]) -> Tuple[List[str], int]:
    """GUIDE.md §5 v2 팬아웃 상한: 이웃을 전부 펼치지 않고 관계 타입 우선으로 최대 FANOUT_CAP개.

    Returns: (선택된 이웃, 잘린 개수)
    """
    g, _, _ = load_graph()
    reqs = set(required_rels or [])
    nbrs = set(g.predecessors(cur)) | set(g.successors(cur))
    scored: List[Tuple[int, int, str, str]] = []
    for nb in nbrs:
        rels: set = set()
        multi = 0
        if g.has_edge(cur, nb):
            for d in g[cur][nb].values():
                rels.add(d.get("relation"))
                multi += 1
        if g.has_edge(nb, cur):
            for d in g[nb][cur].values():
                rels.add(d.get("relation"))
                multi += 1
        match = 1 if (reqs & rels) else (0 if reqs else 0)
        scored.append((-match, -multi, node_label(nb), nb))
    scored.sort()
    ordered = [nb for _, _, _, nb in scored]
    cut = max(0, len(ordered) - FANOUT_CAP)
    return ordered[:FANOUT_CAP], cut


def _bfs_from(starts: List[str], required_rels: Optional[List[str]] = None
              ) -> Tuple[List[Dict], List[str], set, int, int, int]:
    """단일 언급 그룹의 시작 노드들에서 BFS (허브 경유 허용, Movement 1회 제한, 팬아웃 상한).

    반환: paths, order방문노드, 도달집합, 최대홉, 팬아웃절단수, movement차단수.
    """
    g, _, ntype = load_graph()
    required_rels = required_rels or []
    start_set = set(starts)
    paths: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {s: 0 for s in starts}
    reachable = set(starts)
    fanout_cuts = 0
    move_prunes = 0
    maxd = 0
    frontier: List[Tuple[str, List[str], List[Dict], int]] = [
        (s, [s], [], 0) for s in starts]  # node, node경로, edge경로, movement경유수
    for depth in range(1, MAX_HOPS + 1):
        nxt: List[Tuple[str, List[str], List[Dict], int]] = []
        for cur, npath, epath, mov in frontier:
            nbrs, cut = _ranked_neighbors(cur, required_rels)
            fanout_cuts += cut
            for nb in nbrs:
                if nb in npath or nb in start_set:
                    continue
                trips = _edge_triples(cur, nb)
                if not trips:
                    continue
                new_nodes, new_edges = npath + [nb], epath + trips
                nmov = mov + (1 if (ntype.get(nb) == "Movement") else 0)
                if nmov > MAX_MOVEMENT_TRANSIT:
                    move_prunes += 1
                    paths.append({"nodes": new_nodes, "edges": new_edges})
                    reachable.add(nb)
                    maxd = max(maxd, depth)
                    continue
                if nb in seen:
                    continue
                seen[nb] = depth
                reachable.add(nb)
                paths.append({"nodes": new_nodes, "edges": new_edges})
                maxd = max(maxd, depth)
                nxt.append((nb, new_nodes, new_edges, nmov))
                if len(paths) >= MAX_PATHS:
                    break
            if len(paths) >= MAX_PATHS:
                break
        frontier = nxt
        if not frontier or len(paths) >= MAX_PATHS:
            break
    order = sorted(reachable)
    return paths, order, reachable, maxd, fanout_cuts, move_prunes


def node_expand(state: AgentState) -> Dict[str, Any]:
    all_paths, order_all, maxd = [], [], 0
    fanout_cuts = 0
    move_prunes = 0
    reachable: Dict[str, List[str]] = {}
    reqs = state.get("required_rels", [])
    for gi, m in enumerate(state.get("mentions", [])):
        paths, order, reach, d, fc, mp = _bfs_from(m["nodes"], reqs)
        all_paths += paths
        maxd = max(maxd, d)
        fanout_cuts += fc
        move_prunes += mp
        reachable[f"g{gi}"] = sorted(reach)
        order_all += [n for n in order if n not in order_all]
    return {"visited_paths": all_paths[:MAX_PATHS], "visited_nodes": order_all,
            "hops_taken": maxd, "reachable": reachable,
            "fanout_truncations": fanout_cuts,
            "truncated_by_cap": fanout_cuts > 0,
            "movement_prunes": move_prunes}


def _evidence_score(t: Dict[str, Any], reqs: set, starts: set,
                    node_grp: Dict[str, int], on_path: bool = False) -> float:
    s = 0.0
    if on_path:
        s += 5.0  # GUIDE.md §5 v2: 실제 탄 경로 위의 엣지를 최우선
    if t.get("relation") in reqs:
        s += 2.0
    if t.get("subject_id") in starts or t.get("object_id") in starts:
        s += 2.0
    ga, gb = node_grp.get(t.get("subject_id")), node_grp.get(t.get("object_id"))
    if ga is not None and gb is not None and ga != gb:
        s += 3.0
    s += min(t.get("support", 1), 5) * 0.2  # 반복 확인된 근거(다수지지) 우선
    return s


def node_collect_evidence(state: AgentState) -> Dict[str, Any]:
    """GUIDE.md §5 v2 근거 상한: 경로상 엣지 우선으로 최대 EVIDENCE_CAP개."""
    pool: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    nodes = list(state.get("start_entities", [])) + [
        n for p in state.get("visited_paths", []) for n in p["nodes"]]
    for nid in dict.fromkeys(nodes):  # 순서 유지 dedup
        for t in incident_triples(nid):
            pool.setdefault((t["subject_id"], t["relation"], t["object_id"]), t)
    path_keys = {(t["subject_id"], t["relation"], t["object_id"])
                 for p in state.get("visited_paths", []) for t in p["edges"]}
    reqs = set(state.get("required_rels", []))
    starts = set(state.get("start_entities", []))
    groups = _groups_of(state)
    node_grp: Dict[str, int] = {}
    for i, gset in enumerate(groups):
        for n in gset:
            node_grp[n] = i

    def key(t: Dict[str, Any]):
        on = (t["subject_id"], t["relation"], t["object_id"]) in path_keys
        return (-_evidence_score(t, reqs, starts, node_grp, on),
                t["subject_id"], t["relation"] or "", t["object_id"])

    ranked = sorted(pool.values(), key=key)
    ev = ranked[:EVIDENCE_CAP]
    cut = max(0, len(ranked) - len(ev))
    trunc = bool(state.get("truncated_by_cap")) or cut > 0
    return {"evidence_triples": ev, "evidence_truncated": cut,
            "truncated_by_cap": trunc}


def _groups_of(state: AgentState) -> List[set]:
    return [set(m["nodes"]) for m in state.get("mentions", [])]


def node_decide(state: AgentState) -> Dict[str, Any]:
    q = state["question"]
    starts = state.get("start_entities", [])
    ev = state.get("evidence_triples", [])
    unlinked = state.get("unlinked", [])
    if not starts:
        extra = f" 매칭 실패 표기: {', '.join(unlinked)}." if unlinked else ""
        return {"refused": True,
                "refusal_reason": f"시작 개체를 그래프에 연결하지 못했습니다.{extra} "
                                  "aliases.json 역방향 매칭으로 붙는 표기가 없어 억지 연결하지 않았습니다."}
    if not ev:
        return {"refused": True,
                "refusal_reason": f"근거 삼중항이 0개입니다. "
                                  f"시작 개체 {starts}에서 {MAX_HOPS}홉까지 확장했으나 이웃이 없습니다."}
    groups = _groups_of(state)
    connected = True
    if len(groups) >= 2:
        reach = state.get("reachable", {})
        glist = [set(reach.get(f"g{i}", [])) | groups[i] for i in range(len(groups))]
        linked = any(glist[i] & glist[j] for i in range(len(groups)) for j in range(i + 1, len(groups)))
        if not linked:
            ends = {(t["subject_id"], t["object_id"]) for t in ev}
            node_grp = {}
            for i, gset in enumerate(groups):
                for n in gset:
                    node_grp[n] = i
            linked = any(node_grp.get(a) is not None and node_grp.get(a) != node_grp.get(b)
                         and node_grp.get(b) is not None for a, b in ends)
        connected = linked
        if not connected:
            return {"refused": True, "connected": False,
                    "refusal_reason": "찾은 경로가 질문이 요구하는 관계 타입과 맞지 않습니다. "
                                      f"질문 속 서로 다른 언급({len(groups)}개 그룹)이 그래프에서 연결되지 않았습니다: "
                                      + " / ".join("{" + ",".join(sorted(s)) + "}" for s in groups)}
    reqs = state.get("required_rels", [])
    if reqs:
        ev_rels = {t["relation"] for t in ev}
        if not (set(reqs) & ev_rels):
            return {"refused": True, "connected": connected,
                    "refusal_reason": "찾은 경로가 질문이 요구하는 관계 타입과 맞지 않습니다. "
                                      f"질문 요구 관계 {reqs}, 탐색 근거의 관계 {sorted(ev_rels)}."}
    return {"refused": False, "connected": connected, "refusal_reason": ""}


def _llm_client():
    from google import genai
    key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_GENERATIVE_AI_API_KEY is not set")
    return genai.Client(api_key=key)


def _pick_for_llm(state: AgentState) -> List[Dict[str, Any]]:
    ev = state.get("evidence_triples", [])
    reqs = set(state.get("required_rels", []))
    starts = set(state.get("start_entities", []))
    groups = _groups_of(state)
    node_grp = {}
    for i, gset in enumerate(groups):
        for n in gset:
            node_grp[n] = i
    path_keys = {(t["subject_id"], t["relation"], t["object_id"])
                 for p in state.get("visited_paths", []) for t in p["edges"]}

    def score(t):
        on = (t["subject_id"], t["relation"], t["object_id"]) in path_keys
        return _evidence_score(t, reqs, starts, node_grp, on)

    ranked = sorted(range(len(ev)), key=lambda i: (-score(ev[i]), i))
    return [ev[i] for i in ranked[:EVIDENCE_CAP]]


GENERATE_SYS = (
    "너는 디자인사 지식그래프 질의응답기다. 아래에 제시된 증거 삼중항(그래프에서 실제로 탐색한 근거, "
    "출처 문서 포함)과 탐색 경로만으로 답하라. 너의 사전 지식을 쓰지 마라."
)

REFUSAL_PHRASE = "근거를 찾지 못했습니다"


def node_generate(state: AgentState) -> Dict[str, Any]:
    picks = _pick_for_llm(state)
    lines = []
    for i, t in enumerate(picks):
        lines.append(
            f"[{i}] {t['subject']} -{t['relation']}-> {t['object']} "
            f"[지지 {t.get('support', 1)}건] "
            f"(doc: {t['source_doc']}, sent: {(t['source_sentence'] or '')[:200]})")
    path_lines = []
    for p in state.get("visited_paths", [])[:30]:
        segs = []
        for t in p["edges"][:4]:
            segs.append(f"{node_label(t['subject_id'])} -{t['relation']}-> {node_label(t['object_id'])}")
        path_lines.append("  - " + " / ".join(segs) + (" ..." if len(p["edges"]) > 4 else ""))
    reqs = state.get("required_rels", [])
    prompt = (
        f"{GENERATE_SYS}\n\n질문: {state['question']}\n"
        + (f"질문이 요구하는 관계 타입 힌트: {reqs}\n" if reqs else "")
        + "\n[증거 삼중항]\n" + ("\n".join(lines) if lines else "(없음)")
        + "\n\n[탐색 경로(일부)]\n" + ("\n".join(path_lines) if path_lines else "(없음)")
        + "\n\n판정 규칙:\n"
          "1. 질문이 두 개체 사이의 특정 관계를 전제하면(예: 'X가 Y에서 가르친 과목'), "
          "그 관계를 직접 보여주는 삼중항이 증거에 있어야 한다. 없으면 answerable=false.\n"
          "2. 질문이 묻는 대상(예: 자동차, 과목명)이 증거의 어떤 노드와도 대응되지 않으면 "
          "answerable=false. 다른 학교·다른 작품을 끌어와 답하지 마라.\n"
          "3. 증거만으로 답을 특정할 수 없으면 answerable=false.\n"
          "4. 동일 관계의 후보가 여러 개면 지지 건수가 많고 질문의 연결 고리(예: 설립자)와 "
          "이어지는 것을 우선하라. 지지 1건뿐인 삼중항은 추출 노이즈일 수 있으니, "
          "이를 유일한 근거로 답을 특정하지 마라.\n"
          "5. 질문에 중간 연결 고리가 있으면(예: '설립한 사람이 가르친 학교에서 함께 가르친 인물이 "
          "디자인한 작품'), 각 고리를 증거에서 차례로 확인하라: (a) 설립자 확인, (b) 그가 가르친 학교 확인, "
          "(c) 같은 학교에서 가르친 다른 인물 확인, (d) 그 다른 인물의 작품 확인. "
          "답은 마지막 고리(다른 인물)의 삼중항에서만 취하라. 설립자 본인의 작품을 답으로 삼지 마라.\n"
          "출력은 JSON 한 개: "
          '{"answerable": bool, "answer": "...", "used_evidence": [번호], '
          '"used_path": "...", "reason": "..."}. '
          "answer는 한국어로 간결히. answerable=false면 answer는 빈 문자열."
    )
    try:
        from google.genai import types
        client = _llm_client()
        cfg = types.GenerateContentConfig(
            system_instruction=GENERATE_SYS,
            response_mime_type="application/json",
            temperature=0.0,
        )
        resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
        data = json.loads(resp.text)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")
    except Exception as e:  # noqa: BLE001 — LLM 실패는 거절로 전환 (지어내지 않음)
        return {"refused": True, "answer": "",
                "refusal_reason": f"답변 생성용 LLM 호출에 실패했습니다({type(e).__name__}: {str(e)[:150]}). "
                                  "근거만으로 답을 만들 수 없어 거절합니다."}
    if not data.get("answerable"):
        return {"refused": True, "answer": "",
                "llm_reason": str(data.get("reason", "")),
                "refusal_reason": f"{REFUSAL_PHRASE}: {data.get('reason', '증거가 질문을 뒷받침하지 않습니다.')}"}
    used = [picks[i] for i in data.get("used_evidence", []) if isinstance(i, int) and 0 <= i < len(picks)]
    return {"refused": False, "answer": str(data.get("answer", "")),
            "llm_reason": str(data.get("reason", "")),
            "evidence_triples": used + [t for t in state.get("evidence_triples", []) if t not in used]
            if used else state.get("evidence_triples", [])}


def node_refuse(state: AgentState) -> Dict[str, Any]:
    starts = state.get("start_entities", [])
    hops = state.get("hops_taken", 0)
    vnodes = state.get("visited_nodes", [])
    mp = state.get("movement_prunes", 0)
    trunc = bool(state.get("truncated_by_cap", False))
    fc = state.get("fanout_truncations", 0)
    ec = state.get("evidence_truncated", 0)
    base = state.get("refusal_reason", REFUSAL_PHRASE)
    if not base.startswith(REFUSAL_PHRASE):
        base = f"{REFUSAL_PHRASE}: {base}"
    detail = (f" [탐색 현황] 시작 개체: {starts if starts else '연결 실패'} / "
              f"도달 홉: {hops}/{MAX_HOPS} / 방문 노드 {len(vnodes)}개"
              + (f": {vnodes[:15]}" + (" ..." if len(vnodes) > 15 else "") if vnodes else "")
              + f" / 근거 삼중항 {len(state.get('evidence_triples', []))}개"
              + (f" / 상한 절단: 팬아웃 {fc}건·근거 {ec}건" if trunc else " / 상한 절단 없음")
              + (f" / Movement 경유 제한 {mp}건" if mp else ""))
    return {"answer": base + detail, "refused": True, "refusal_reason": base + detail}


def build_agent():
    builder = StateGraph(AgentState)
    builder.add_node("link_entities", node_link_entities)
    builder.add_node("expand", node_expand)
    builder.add_node("collect_evidence", node_collect_evidence)
    builder.add_node("decide", node_decide)
    builder.add_node("generate", node_generate)
    builder.add_node("refuse", node_refuse)
    builder.add_edge(START, "link_entities")
    builder.add_conditional_edges(
        "link_entities",
        lambda s: "decide" if not s.get("start_entities") else "expand",
        {"expand": "expand", "decide": "decide"})
    builder.add_edge("expand", "collect_evidence")
    builder.add_edge("collect_evidence", "decide")
    builder.add_conditional_edges(
        "decide", lambda s: "refuse" if s.get("refused") else "generate",
        {"refuse": "refuse", "generate": "generate"})
    builder.add_conditional_edges(
        "generate", lambda s: "refuse" if s.get("refused") else END,
        {"refuse": "refuse", END: END})
    builder.add_edge("refuse", END)
    return builder.compile()


_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def answer_question(question: str) -> dict:
    """진입점 (W4/W5 import용). 질문 -> 결과 dict."""
    load_graph()
    out = get_agent().invoke({"question": question})
    return {
        "question": question,
        "start_entities": out.get("start_entities", []),
        "hops_taken": out.get("hops_taken", 0),
        "visited_paths": out.get("visited_paths", []),
        "evidence_triples": out.get("evidence_triples", []),
        "answer": out.get("answer", ""),
        "refused": bool(out.get("refused", False)),
        "refusal_reason": out.get("refusal_reason", ""),
        "truncated_by_cap": bool(out.get("truncated_by_cap", False)),
        "fanout_truncations": out.get("fanout_truncations", 0),
        "evidence_truncated": out.get("evidence_truncated", 0),
        "movement_prunes": out.get("movement_prunes", 0),
        # 하위 호환 (v1 필드, 항상 비어 있음 — GUIDE.md §5 v2에서 허브 금지 폐기)
        "blocked_by_hub": False,
        # W4/W5 및 디버깅용 부가 정보
        "visited_nodes": out.get("visited_nodes", []),
        "hub_blocked": [],
        "mentions": out.get("mentions", []),
    }


def run_goldenset() -> List[dict]:
    """goldenset 12문항 실행 -> output/runs.jsonl."""
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    rows = []
    for item in data["items"]:
        try:
            r = answer_question(item["question"])
        except Exception as e:  # noqa: BLE001 — 한 문항 실패가 전체를 막지 않게
            r = {"question": item["question"], "start_entities": [], "hops_taken": 0,
                 "visited_paths": [], "evidence_triples": [], "answer": "",
                 "refused": True, "refusal_reason": f"에이전트 오류: {type(e).__name__}: {e}",
                 "truncated_by_cap": False, "fanout_truncations": 0,
                 "evidence_truncated": 0, "movement_prunes": 0,
                 "blocked_by_hub": False, "visited_nodes": [], "hub_blocked": [],
                 "mentions": [], "error": str(e)[:300]}
        r = {"qid": item.get("id"), **r}
        rows.append(r)
        print(f"[{r['qid']}] refused={r['refused']} hops={r['hops_taken']} "
              f"starts={r['start_entities']} ev={len(r['evidence_triples'])} "
              f"trunc={r.get('truncated_by_cap')} ans={(r['answer'] or '')[:80]}", flush=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {RUNS_PATH} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    try:
        d = get_agent().get_graph().draw_mermaid()
        DIAGRAM_PATH.write_text(d, encoding="utf-8")
        print(f"wrote {DIAGRAM_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"diagram skipped: {e}")
    run_goldenset()
