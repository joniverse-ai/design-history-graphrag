# W4 평가 보고서 (GraphRAG vs basic RAG, 홉수별 분리)

- 모델: gemini-3.1-flash-lite / 베이스라인: BM25 top-5 텍스트 청크, 그래프 미사용
- 채점: 결정적 키워드 판정(엄격). LLM 심판 미사용.
- 전체 평균은 내지 않는다(홉수별 분리 보고가 과제 기준).

## 1. 홉수별 정답률 (GraphRAG vs basic RAG)

| 버킷 | n | GraphRAG 정답 | basic RAG 정답 | 격차(G−B) |
|---|---|---|---|---|
| 1hop | 2 | 2/2 (100%) | 1/2 (50%) | +50% |
| 2hop | 5 | 5/5 (100%) | 2/5 (40%) | +60% |
| 3hop | 3 | 3/3 (100%) | 0/3 (0%) | +100% |
| refusal | 2 | 2/2 (100%) | 2/2 (100%) | +0% |

## 2. 경로 재현율 / 정밀도 (홉수별, GraphRAG)

| 버킷 | 평균 재현율 | 평균 정밀도 | 비고 |
|---|---|---|---|
| 1hop | 1.000 | 0.087 | 전부 3홉·근거 40개(상한) |
| 2hop | 1.000 | 0.100 | 전부 3홉·근거 40개(상한) |
| 3hop | 0.722 | 0.067 | 전부 3홉·근거 40개(상한) |
| refusal | 측정 불가(기대 경로 없음) | 측정 불가(기대 경로 없음) | 전부 3홉·근거 40개(상한) |

## 3. 문항별 상세 (GraphRAG)

| qid | 판정 | 재현율 | 정밀도 | 답변 | 실패층 |
|---|---|---|---|---|---|
| H1-01 | correct | 1.00 | 0.15 | 발터 그로피우스(Walter Gropius) | none |
| H1-02 | correct | 1.00 | 0.03 | 마르셀 브로이어 | none |
| H2-01 | correct | 1.00 | 0.05 | 뉴 바우하우스(New Bauhaus) | none |
| H2-02 | correct | 1.00 | 0.05 | 하버드 대학교 | none |
| H2-03 | correct | 1.00 | 0.05 | 바우하우스, 하버드 대학교 | none |
| H2-04 | correct | 1.00 | 0.30 | 피트 몬드리안, 빌모스 후사르, 바르트 판 데르 레크, J.J.P. 아우트(야코뷔스 아우트), 얀 빌스, 로 | none |
| H2-05 | correct | 1.00 | 0.05 | 일리노이 기술대학(IIT) | none |
| H3-01 | correct | 1.00 | 0.10 | 뉴 바우하우스(New Bauhaus) | none |
| H3-02 | correct | 0.67 | 0.05 | Cesca Chair, Long Chair, Wassily Chair | none |
| H3-03 | correct | 0.50 | 0.05 | 바실리 의자(Wassily Chair), 체스카 의자(Cesca Chair) | none |
| R-01 | correct | - | - | 근거를 찾지 못했습니다: 제시된 증거 삼중항과 탐색 경로를 검토한 결과, 마르셀 브로이어가 디자인한 자동차에 | none |
| R-02 | correct | - | - | 근거를 찾지 못했습니다: 찾은 경로가 질문이 요구하는 관계 타입과 맞지 않습니다. 질문 속 서로 다른 언급( | none |

## 4. 문항별 상세 (basic RAG)

| qid | 판정 | 답변 | 실패층 | 실패 근거 |
|---|---|---|---|---|
| H1-01 | correct | 발터 그로피우스 | none | 정답이므로 실패 층 없음. |
| H1-02 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (브로이어, breuer)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| H2-01 | correct | 뉴 바우하우스 | none | 정답이므로 실패 층 없음. |
| H2-02 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (하버드, harvard)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| H2-03 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (브로이어, breuer)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| H2-04 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (리트벨트, rietveld)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| H2-05 | correct | 일리노이 공과대학 | none | 정답이므로 실패 층 없음. |
| H3-01 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (브로이어, breuer)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| H3-02 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (브로이어, breuer)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| H3-03 | wrong |  | retrieval | 검색된 top-k 청크에 답의 핵심 실마리 (브로이어, breuer)가 없음 -> basic RAG 검색(retrieval) 실패로 귀속. |
| R-01 | correct |  | none | 정답이므로 실패 층 없음. |
| R-02 | correct |  | none | 정답이므로 실패 층 없음. |

## 5. 실패 층 분류 (GraphRAG, 오답만 귀속)

- 오답 0건: 귀속할 실패 없음(현행 12문항·엄격 기준).

## 5b. Grounding 미세 결함 (정답이나 기대 경로 일부가 근거에 없음)

- H3-02: 정답이나 기대 경로 일부가 근거 40개에 없음(상한 절단). 답은 아래 우회 근거/원문 문장으로 도출됨: Institution(바우하우스) <-TAUGHT_AT- Person(브로이어) | 우회 근거: 문장 언급(en_Alan_I_W_Frank_House.md): The Alan I W Frank House is a private residence in Pittsburgh, Pennsylvania, designed by B…
- H3-03: 정답이나 기대 경로 일부가 근거 40개에 없음(상한 절단). 답은 아래 우회 근거/원문 문장으로 도출됨: Institution(하버드 GSD) <-TAUGHT_AT- Person(브로이어) | 우회 근거: Marcel Breuer -TAUGHT_AT-> Bauhaus // Person(브로이어) -DESIGNED-> Work(바실리 체어) | 우회 근거: Marcel Breuer -DESIGNED-> Cesca Chair

## 6. 색인 결함 목록 (그래프 실측 기반)

### D1. en_Weimar.md발 FOUNDED 오추출 4건 (알려진 결함, 그래프 잔류 확인)
- 층: index
  - `Wassily Kandinsky -FOUNDED-> Bauhaus (doc: en_Weimar.md)`
  - `Paul Klee -FOUNDED-> Bauhaus (doc: en_Weimar.md)`
  - `Henry van de Velde -FOUNDED-> Bauhaus (doc: en_Weimar.md)`
  - `Lyonel Feininger -FOUNDED-> Bauhaus (doc: en_Weimar.md)`
- source_sentence: Later, artists and architects including Henry van de Velde, Wassily Kandinsky, Paul Klee, Lyonel Feininger, and Walter Gropius came to the city and founded the Bauhaus movement, the most important German design school of the interwar period.
- why_wrong: 원문은 'came to the city and founded the Bauhaus movement' — 사조(Movement)에 대한 서술을 학교(Institution) 설립으로 추출했고, 실제로 마이스터로 부임한 4인을 설립자로 만들었다. Gropius의 FOUNDED는 별도 근거로 정당하므로 제외.
- impact: H1-01은 Gropius 삼중항 덕에 정답이 나오지만, '바우하우스 설립자' 질의의 근거 정밀도를 갉아먹는 색인층 결함이다.

### D2. FOUNDED 주어 오인 + 동일 학교 3중복 — van de Velde 관련 Institution 노드
- 층: index
  - `Henry van de Velde -FOUNDED-> Grand-Ducal School of Arts and Crafts (doc: en_Art_Nouveau.md)`
  - `Henry van de Velde -FOUNDED-> Grand Ducal Saxon School of Arts and Crafts (doc: en_Bauhaus.md)`
- source_sentence: In 1906, he departed Belgium for Weimar (Germany), where he founded the Grand-Ducal School of Arts and Crafts, where the teaching of historical styles was forbidden.
- why_wrong: (a) en_Bauhaus.md 삼중항은 명백한 주어 오인이다. 원문 주어는 'the Grand Duke of Saxe-Weimar-Eisenach'이고 van de Velde는 'directed by'의 대상(교장)이다. 수동문 주어를 설립자로 오인한 추출 오류. (b) en_Art_Nouveau.md 삼중항('he founded the Grand-Ducal School')은 문장 자체는 van de Velde 설립을 서술하나, 결과적으로 같은 학교가 'Grand-Ducal School of Arts and Crafts' / 'Grand Ducal Saxon School of Arts and Crafts' / 'Kunstgewerbeschule Weimar' 3개 Institution 노드로 중복됐다. 별칭 병합 실패(색인층)다.
- impact: 설립자 질의에서 오답 후보를 만들고, 동일 학교가 3개 노드로 분단돼 TAUGHT_AT/FOUNDED 경로 탐색을 분산시킨다.

### D3. 역사 서술 라벨의 Institution 오분류 — 'Second School of Chicago'
- 층: index
  - `Ludwig Mies van der Rohe -FOUNDED-> Second School of Chicago (node_type=Institution, doc: en_International_Style.md)`
- source_sentence: When Mies fled in 1938, he first fled to England, but on emigrating to the US he went to Chicago, founded the Second School of Chicago at IIT and solidified his reputation as a prototypical modern architect.
- why_wrong: 'Second School of Chicago'는 건축사 서술상의 유파 라벨이지 설립 가능한 학교/회사가 아니다. 스키마(FOUNDED: Person->Institution|Company) 위반.
- impact: Mies 관련 설립 질의의 정밀도를 떨어뜨리는 색인층 결함.

### D4. 예술 그룹의 Company 오분류 — '네오모던' (FOUNDED 대상 오용)
- 층: index
  - `가이 데닝 -FOUNDED-> 네오모던 (node_type=Company, doc: ko_네오모던.md)`
- source_sentence: 네오모던 예술가 그룹은 1997년 가이 데닝(Guy Denning)에 의해 창립되었으며, 현대 예술의 다양성이 국가 지원을 받는 예술 기관과 단체들에 의해 억압받고 있다는 전제하에 시작되었다.
- why_wrong: '네오모던 예술가 그룹'은 운동/집단이지 회사가 아니다. FOUNDED의 대상은 Institution|Company여야 하므로 스키마 위반.
- impact: FOUNDED 관계 전체의 신뢰도를 갉아먹는 추출 오류.

### D5. 일반명사 Work 노드 약 60개 잔류 후보 (GUIDE §4.4 위반)
- 층: index
  - 규모: 60개
    - products (deg 3)
    - art nouveau designs (deg 2)
    - costumes and sets (deg 2)
    - houses (deg 2)
    - a floor (deg 1)
    - a massive fireplace (deg 1)
    - alphabet (deg 1)
    - balustrades (deg 1)
    - bottles (deg 1)
    - bridges and service stations (deg 1)
    - building (deg 1)
    - buildings (deg 1)
    - clothing (deg 1)
    - crematorium (deg 1)
    - dresses (deg 1)
- why_wrong: 'building', 'houses', 'clothing', 'jewelry', 'products', 'art nouveau designs', 'costumes and sets' 등 일반명사가 Work 노드로 추출됐다. §4에서 제외하기로 한 것들이다. 집계는 소문자 시작·일반명사형만 센 엄격 기준이며, 소문자 스타일 정식 명칭이 극소수 섞여 있을 수 있다.
- impact: 대부분 deg 1이라 이번 12문항 채점에는 직접 영향이 없으나, DESIGNED 관계(935건)의 정밀도를 구조적으로 떨어뜨린다.

### CHECK-R. 거절 문항의 그래프 내 뒷받침 존재 여부 (오답이 아닌지 확인용)
- 층: index
- why_wrong: 해당 없음(확인용 항목).
- impact: 두 값 모두 0이어야 거절이 정당하다. 0이 아니면 거절 문항 설계 자체가 깨진다. 실측: Breuer-자동차 0건, Rams-TAUGHT_AT-Bauhaus 0건 -> 거절 정당.
- breuer_car_designed_triples: 0
- rams_taught_at_bauhaus_triples: 0

## 7. 막힌 지점 / 한계
- 조기 종료 없음: 답을 찾아도 3홉까지 확장하므로 12문항 전부 hops_taken=3, 근거 40개(상한)다. 정밀도 하락의 직접 원인. GUIDE §7 알려진 한계.
- 거절 문항의 경로 재현율/정밀도는 기대 경로가 없어 측정 불가(null).
- 기본 RAG 청크 분할(~700자)·토크나이저(영숫자/한글 블록)는 단순 구현이며, 임베딩 기반 검색이 아니라는 한계가 있다.
