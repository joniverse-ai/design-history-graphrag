# 디자인사 GraphRAG 에이전트

한국 영화 GraphRAG 수업을 **디자인사(Design History)** 도메인으로 옮긴 멀티홉 질의응답 에이전트.
시작 개체 → 최대 3홉 확장 → 실제 경로 기록 → 근거 삼중항으로만 답변하고, 근거가 없으면
지어내지 않고 **"근거를 찾지 못했습니다"** 와 함께 탐색 현황을 보여준다.

## 필요한 환경변수

| 변수 | 용도 |
|---|---|
| `GOOGLE_GENERATIVE_AI_API_KEY` | 답변 생성용 Gemini 호출 (`agent.py`) — **키는 코드·저장소에 넣지 않고 `os.environ` 으로만 읽는다** |

## 설치

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` (핵심):

```
streamlit
networkx
langgraph
google-genai
pandas
```

## 실행 순서

| 순서 | 스크립트 | 용도 | 명령 |
|---|---|---|---|
| 1 (W1) | `scripts/collect_corpus.py` | 한/영 위키 코퍼스 수집 (채택·탈락 로그) | `python scripts/collect_corpus.py` |
| 2 (W2) | `build_graph.py` | 스키마 제한 추출 + 정규화 → `output/graph.graphml` | `python build_graph.py` |
| 3 (W3) | `agent.py` | 시작 개체→3홉 확장→경로 기록→근거 답변/거절. 골든셋 12문항 실행 → `output/runs.jsonl` | `python agent.py` |
| 4 (W4) | `evaluate.py` | 홉수별 채점(경로 재현율·정밀도) + basic RAG 대조 + 실패 3층 분류(색인/탐색/생성) | `python evaluate.py` |
| 5 (W5) | `app.py` | Streamlit 데모 (답변 + 경로 + 삼중항 + 출처) | `streamlit run app.py` |

각 단계의 정책 근거(홉 상한 3, 허브 허용, 팬아웃·근거 상한, 거절 조건)는 `GUIDE.md` §5를 따른다.

## 데모 실행법

```bash
export GOOGLE_GENERATIVE_AI_API_KEY="..."
streamlit run app.py
```

브라우저에서 질문을 입력하거나 예시 질문(골든셋 12문항) 버튼을 눌러 바로 실행할 수 있다.
응답 화면: **답변 → 실제 탐색 경로(서브그래프 + `A ──REL──> B` 나열) → 근거 삼중항(표) →
출처 문서·문장(expander)**. 거절 시에는 답변 대신 **"근거를 찾지 못했습니다"** 와
방문 노드·홉 수·절단 사유를 보여준다.

## 저장소 구조

```
.
├── GUIDE.md                 # 워커 공통 기준 (스키마·코퍼스·멀티홉 정책·골든셋·평가 설계)
├── app.py                   # W5 Streamlit 데모 (answer_question import, 수정 금지 파일 아님)
├── agent.py                 # W3 에이전트 진입점 answer_question() — 수정 금지
├── build_graph.py           # W2 스키마 제한 추출 + 정규화 — 수정 금지
├── data/
│   ├── docs/                # 원문 코퍼스 70건 (한/영 위키)
│   ├── goldenset.json       # 골든셋 12문항 (1홉 2 · 2홉 5 · 3홉 3 · 거절 2)
│   ├── aliases.json         # 한영 별칭 사전 (개체 연결용)
│   └── manifest.json        # 코퍼스 채택·탈락 로그
└── output/
    ├── graph.graphml        # 지식그래프 (1299 노드 / 1311 엣지) — 수정 금지
    ├── runs.jsonl           # 골든셋 12문항 실행 기록 (반환 dict 실제 스키마 확인용)
    ├── triples.jsonl        # 정규화 전 원시 삼중항
    ├── merge_log.json       # 별칭 병합 로그
    ├── stats.json           # 그래프 통계
    └── agent_architecture.mmd  # LangGraph 구조 다이어그램
```
