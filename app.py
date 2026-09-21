#!/usr/bin/env python3
"""W5: 디자인사 GraphRAG Streamlit 데모.

- agent.py 의 answer_question(question) 을 그대로 import 해 쓴다 (수정 없음).
- 그래프 로딩/에이전트 초기화는 @st.cache_resource 로 1회만 수행.
- 화면: 질문 입력 + 예시 질문(골든셋 12문항) / 답변 / 실제 탐색 경로 /
  근거 삼중항(표) / 출처 문서·문장(expander). 거절 시에는
  "근거를 찾지 못했습니다"와 탐색 현황(방문 노드·홉 수·절단 사유)을 보여준다.

실행: streamlit run app.py
키: GOOGLE_GENERATIVE_AI_API_KEY (환경변수로만, 코드에 넣지 않음)
"""

import json
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = ROOT / "data" / "goldenset.json"

MAX_PATHS_SHOWN = 10      # 경로 나열 상한 (읽기용)
MAX_GRAPH_EDGES = 20      # 서브그래프에 그릴 엣지 상한
MAX_TRIPLES_TABLE = 40    # 근거 상한과 동일 (GUIDE.md §5 EVIDENCE_CAP)


# ---------------------------------------------------------------- 초기화 (캐시)

@st.cache_resource
def get_agent():
    """그래프 로딩 + LangGraph 에이전트 컴파일. 질문마다 다시 로드하지 않는다."""
    import agent as _agent

    _agent.load_graph()
    _agent.get_agent()
    return _agent


@st.cache_data
def load_goldenset():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["items"]


# ---------------------------------------------------------------- 표시 헬퍼

def short_id(nid: str) -> str:
    """'Person:walter gropius' -> 'Walter Gropius' 식으로 짧게. 라벨이 있으면 라벨 우선."""
    return nid.split(":", 1)[-1] if ":" in nid else nid


def format_path(path: dict) -> str:
    """방문 경로를 'A ──REL──> B ──REL──> C' 문자열로.

    한 스텝에 병렬 엣지가 함께 기록되므로(_edge_triples), 이어지지 않는 엣지는
    ' / ' 로 구분해 잘못된 체인처럼 읽히지 않게 한다.
    """
    edges = path.get("edges", [])
    if not edges:
        nodes = path.get("nodes", [])
        return " → ".join(short_id(n) for n in nodes) if nodes else "(빈 경로)"
    segs: list[str] = []
    prev_oid, prev_label = None, None
    for t in edges:
        s = t.get("subject") or short_id(t.get("subject_id", "?"))
        o = t.get("object") or short_id(t.get("object_id", "?"))
        rel = t.get("relation", "?")
        if prev_oid is not None and (t.get("subject_id") == prev_oid or s == prev_label):
            segs.append(f"──{rel}──> {o}")
        else:
            if segs:
                segs.append("  /  ")
            segs.append(f"{s} ──{rel}──> {o}")
        prev_oid, prev_label = t.get("object_id"), o
    return "".join(segs)


def build_graphviz(result: dict) -> str | None:
    """상위 경로들에서 작은 서브그래프를 DOT 로 만든다. 노드/엣지 수 제한."""
    edges = []
    seen = set()
    for p in result.get("visited_paths", []):
        for t in p.get("edges", []):
            key = (t.get("subject_id"), t.get("relation"), t.get("object_id"))
            if key in seen:
                continue
            seen.add(key)
            edges.append(t)
            if len(edges) >= MAX_GRAPH_EDGES:
                break
        if len(edges) >= MAX_GRAPH_EDGES:
            break
    if not edges:
        return None
    starts = set(result.get("start_entities", []))

    def esc(s: str) -> str:
        return (s or "?").replace('"', "'").replace("\n", " ")

    lines = ["digraph {", '  rankdir=LR; node [shape=box, fontsize=10];']
    nodes = {}
    for t in edges:
        for nid_key, label_key in (("subject_id", "subject"), ("object_id", "object")):
            nid = t.get(nid_key, "?")
            if nid not in nodes:
                label = esc(t.get(label_key) or short_id(nid))
                color = "lightblue" if nid in starts else "white"
                nodes[nid] = f'  "{esc(nid)}" [label="{label}", style=filled, fillcolor={color}];'
    lines += list(nodes.values())
    for t in edges:
        lines.append(
            f'  "{esc(t.get("subject_id", "?"))}" -> "{esc(t.get("object_id", "?"))}"'
            f' [label="{esc(t.get("relation", "?"))}", fontsize=9];'
        )
    lines.append("}")
    return "\n".join(lines)


def render_result(result: dict):
    refused = result.get("refused", False)
    hops = result.get("hops_taken", 0)
    vnodes = result.get("visited_nodes", [])
    paths = result.get("visited_paths", [])
    ev = result.get("evidence_triples", [])
    trunc = bool(result.get("truncated_by_cap", False))

    # ---- 요약 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("도달 홉 수", f"{hops} / 3")
    c2.metric("방문 노드", len(vnodes))
    c3.metric("탐색 경로", len(paths))
    c4.metric("근거 삼중항", len(ev))
    if trunc:
        st.caption(
            f"⚠️ 상한으로 절단됨 (실패 아님): 팬아웃 절단 {result.get('fanout_truncations', 0)}건 · "
            f"근거 절단 {result.get('evidence_truncated', 0)}건 · "
            f"Movement 경유 제한 {result.get('movement_prunes', 0)}건"
        )

    # ---- 답변 / 거절
    st.subheader("💬 답변")
    if refused:
        st.error("근거를 찾지 못했습니다")
        st.write(result.get("refusal_reason") or result.get("answer") or "")
        st.subheader("🔍 어디까지 탐색했나")
        st.write(f"- 시작 개체: `{result.get('start_entities') or '연결 실패'}`")
        st.write(f"- 도달 홉: **{hops} / 3**")
        st.write(f"- 방문 노드 **{len(vnodes)}개**")
        if vnodes:
            with st.expander("방문 노드 목록 보기", expanded=False):
                st.write(", ".join(f"`{n}`" for n in vnodes[:60])
                         + (" …" if len(vnodes) > 60 else ""))
        cut_msg = ("팬아웃·근거 상한에 걸려 잘림" if trunc else "상한 절단 없음")
        st.write(f"- 절단 사유: {cut_msg}")
    else:
        st.success(result.get("answer", "(빈 답변)"))

    # ---- 실제 탐색 경로
    st.subheader("🛤️ 실제로 탄 경로")
    starts = result.get("start_entities", [])
    if starts:
        st.caption("시작 개체: " + " · ".join(f"`{s}`" for s in starts))
    if not paths:
        st.info("기록된 경로가 없습니다.")
    else:
        dot = build_graphviz(result)
        if dot:
            try:
                st.graphviz_chart(dot, use_container_width=True)
            except Exception as e:  # noqa: BLE001 — 그래프 렌더 실패해도 텍스트 경로는 보여준다
                st.caption(f"(서브그래프 렌더 생략: {type(e).__name__})")
        st.caption(f"대표 경로 {min(len(paths), MAX_PATHS_SHOWN)}개 / 전체 {len(paths)}개 "
                   "(긴 경로·전량은 runs.jsonl 참조)")
        for i, p in enumerate(paths[:MAX_PATHS_SHOWN]):
            st.code(f"[경로 {i+1}] {format_path(p)}", language=None)

    # ---- 근거 삼중항 (표)
    st.subheader("📊 근거 삼중항")
    if not ev:
        st.info("근거 삼중항이 없습니다 (0개).")
    else:
        rows = [{
            "주어(subject)": t.get("subject", ""),
            "관계(relation)": t.get("relation", ""),
            "목적어(object)": t.get("object", ""),
            "지지": t.get("support", 1),
        } for t in ev]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # ---- 출처 문서·문장 (접기)
        st.subheader("📚 출처 문서")
        st.caption("각 삼중항이 어느 문서의 어느 문장에서 나왔는지 확인하세요.")
        with st.expander(f"출처 문장 모두 보기 ({len(ev)}건)", expanded=False):
            for i, t in enumerate(ev):
                st.markdown(
                    f"**[{i}]** {t.get('subject')} —{t.get('relation')}→ {t.get('object')}  "
                    f"(지지 {t.get('support', 1)}건)\n\n"
                    f"- 문서: `{t.get('source_doc', '?')}`\n"
                    f"- 문장: {t.get('source_sentence') or '(없음)'}"
                )
                if i < len(ev) - 1:
                    st.divider()


# ---------------------------------------------------------------- UI

st.set_page_config(page_title="디자인사 GraphRAG 데모", layout="wide")
st.title("🏛️ 디자인사 GraphRAG 데모")
st.caption(
    "시작 개체 → 최대 3홉 확장 → 실제 경로 기록 → 근거 삼중항으로만 답변. "
    "근거가 없으면 지어내지 않고 거절합니다. (GUIDE.md §5)"
)

if not os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY"):
    st.warning("⚠️ 환경변수 GOOGLE_GENERATIVE_AI_API_KEY 가 비어 있습니다. "
               "답변 생성(LLM) 단계에서 거절로 전환될 수 있습니다.")

items = load_goldenset()

# ---- 예시 질문 버튼 (골든셋 12문항)
st.subheader("📌 예시 질문 (골든셋 12문항)")
if "question" not in st.session_state:
    st.session_state.question = items[0]["question"]

cols = st.columns(3)
for i, it in enumerate(items):
    tag = {"1": "1홉", "2": "2홉", "3": "3홉"}.get(str(it.get("hops")), "거절")
    label = f"[{it['id']}·{tag}] {it['question'][:26]}…"
    with cols[i % 3]:
        if st.button(label, key=f"ex_{it['id']}", help=it["question"]):
            st.session_state.question = it["question"]

# ---- 질문 입력창
question = st.text_area("✏️ 질문을 입력하세요", key="question", height=70)
run = st.button("🔎 질문하기", type="primary")

if run:
    q = (question or "").strip()
    if not q:
        st.warning("질문을 입력해 주세요.")
    else:
        agent = get_agent()  # 캐시됨: 첫 1회만 로딩
        with st.spinner("그래프를 탐색하고 근거를 모으는 중… (최대 3홉)"):
            try:
                result = agent.answer_question(q)
            except Exception as e:  # noqa: BLE001 — 화면이 죽지 않게
                st.error(f"에이전트 실행 중 오류: {type(e).__name__}: {e}")
                result = None
        if result is not None:
            render_result(result)
