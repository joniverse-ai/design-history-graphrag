#!/usr/bin/env python3
"""W2: schema-restricted triple extraction (Google Gemini) + normalization -> graph.graphml.

Outputs:
  output/triples.jsonl    raw triples pre-normalization (resume-capable, append)
  output/progress.json    done chunk ids (resume)
  output/graph.graphml    networkx MultiDiGraph (edge attrs: relation, source_doc, source_sentence)
  output/merge_log.json   variant -> canonical merges
  data/aliases.json       manual seed + auto-discovered variants (updated in place)
  output/stats.json       counts for reporting

Env: GOOGLE_GENERATIVE_AI_API_KEY (never hardcode).
Model chain: gemini-2.5-flash -> gemini-3.1-flash-lite -> gemini-3.6-flash
(first that works; 429/503 -> exponential backoff; 404 -> next model).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "data" / "docs"
ALIASES_PATH = ROOT / "data" / "aliases.json"
OUT = ROOT / "output"
TRIPLES_PATH = OUT / "triples.jsonl"
PROGRESS_PATH = OUT / "progress.json"
GRAPH_PATH = OUT / "graph.graphml"
MERGE_LOG_PATH = OUT / "merge_log.json"
STATS_PATH = OUT / "stats.json"
MODEL_USED_PATH = OUT / "model_used.json"

MODEL_CHAIN = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]
CHUNK_CHARS = 9000
SLEEP_BETWEEN_CALLS = 2.0
MAX_RETRIES_429 = 6

NODE_TYPES = {"Person", "Movement", "Institution", "Work", "Company"}
RELATIONS = {"STUDIED_AT", "TAUGHT_AT", "FOUNDED", "BELONGS_TO",
             "DESIGNED", "MANUFACTURED_BY", "INFLUENCED_BY"}
# relation -> (allowed subject types, allowed object types)
REL_TYPE_CONSTRAINTS = {
    "STUDIED_AT": ({"Person"}, {"Institution"}),
    "TAUGHT_AT": ({"Person"}, {"Institution"}),
    "FOUNDED": ({"Person"}, {"Institution", "Company"}),
    "BELONGS_TO": ({"Person", "Work"}, {"Movement"}),
    "DESIGNED": ({"Person"}, {"Work"}),
    "MANUFACTURED_BY": ({"Work"}, {"Company"}),
    "INFLUENCED_BY": ({"Person", "Movement"}, {"Person", "Movement"}),
}

# Exact names dropped at normalization (generic/era terms the LLM typed as nodes)
DROP_EXACT = {"세기말", "fin de siècle", "fin de siecle"}
# Lowercase common nouns the LLM typed as Work (exact match, case-insensitive)
WORK_GENERIC_EXACT = {"desk", "desks", "vase", "vases", "pitcher", "lamp", "lamps"}
GENERIC_BLOCK = {
    "디자인", "건축", "예술", "가구", "의자", "추상미술", "그래픽 디자인",
    "공업 디자인", "산업 디자인", "모더니즘", "도시 계획가", "도시계획가",
    "미술", "공예", "사진", "회화", "조각", "음악", "오페라", "타이포그래피",
    "실내 디자인", "도시 계획", "도시계획", "패션", "영화",
    "design", "architecture", "art", "arts", "furniture", "chair", "chairs",
    "abstract art", "graphic design", "industrial design", "modernism",
    "urban planner", "urban planning", "painting", "sculpture", "music",
    "opera", "photography", "typography", "craft", "crafts", "city", "cities",
    "country", "germany", "독일", "미국", "프랑스",
}
# Non-designers that leaked into corpus (composers etc.): never Person nodes
NON_DESIGNER_BLOCK = {
    "alban berg", "alban maria johannes berg", "arnold schoenberg",
    "arnold schonberg", "anton webern", "gustav mahler",
    "알반 베르크", "알반 마리아 요하네스 베르크", "아르놀트 쇤베르크",
    "쇤베르크", "안톤 베베른", "베베른", "아르놀트 쇤베르크와",
}
# Work-typed names that are generic descriptions, not citable works
WORK_GENERIC_SUBSTR = {
    "furniture designs", "modernist", "his designs", "her designs",
    "several works", "many works", "various works", "numerous works",
    "design work", "design works", "his work", "early work",
}

SYSTEM_PROMPT = """You extract design-history knowledge triples from a Wikipedia excerpt.
Return ONLY JSON matching the schema: {"triples": [{subject, subject_type, relation, object, object_type, source_sentence}]}.

Node types (5 ONLY): Person, Movement, Institution, Work, Company.
Relations (7 ONLY, with strict endpoint types):
- STUDIED_AT: Person -> Institution (studied at school)
- TAUGHT_AT: Person -> Institution (taught/lectured at school)
- FOUNDED: Person -> Institution or Company (founded school/company)
- BELONGS_TO: Person or Work -> Movement (member of art/design movement)
- DESIGNED: Person -> Work (designed a work/product/typeface/building)
- MANUFACTURED_BY: Work -> Company (produced by manufacturer)
- INFLUENCED_BY: Person or Movement -> Person or Movement (influenced by)

STRICT RULES:
1. Person = designer, architect, or design educator ONLY. NEVER add composers, musicians, painters-only, writers, or politicians (e.g. Alban Berg, Schoenberg, Webern are composers: EXCLUDE).
2. NEVER create nodes for generic nouns: design, architecture, art, furniture, chair, graphic design, abstract art, modernism (generic use), urban planner, countries, cities, years.
3. NEVER use birthplace/nationality/year relations. Only the 7 relations above.
4. Bauhaus-the-school is Institution; Bauhaus-the-style is Movement. Keep them distinct: if the sentence is about the school as organization use Institution, about style/movement use Movement.
5. source_sentence MUST be a verbatim sentence from the excerpt supporting the triple. One triple per grounding sentence; do not merge evidence across sentences.
6. Use names as written in the excerpt (Korean stays Korean, English stays English). Do NOT translate names.
7. If no valid triple exists, return {"triples": []}.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "triples": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject": {"type": "STRING"},
                    "subject_type": {"type": "STRING"},
                    "relation": {"type": "STRING"},
                    "object": {"type": "STRING"},
                    "object_type": {"type": "STRING"},
                    "source_sentence": {"type": "STRING"},
                },
                "required": ["subject", "subject_type", "relation",
                             "object", "object_type", "source_sentence"],
            },
        }
    },
    "required": ["triples"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def chunk_text(text: str, limit: int = CHUNK_CHARS):
    """Split into <=limit char chunks at paragraph boundaries, 1-para overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) > limit and cur:
            chunks.append("\n\n".join(cur))
            # overlap: carry last paragraph forward
            cur, cur_len = [cur[-1]], len(cur[-1])
        cur.append(p)
        cur_len += len(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


class Extractor:
    def __init__(self):
        from google import genai  # noqa: PLC0415 (lazy: only needed for extract)
        from google.genai import types  # noqa: PLC0415
        key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_GENERATIVE_AI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.types = types
        self.working_model = None
        self.model_used = {}  # model -> n calls

    def _call(self, model, chunk):
        cfg = self.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            temperature=0.0,
        )
        resp = self.client.models.generate_content(
            model=model, contents="Excerpt (Wikipedia):\n\n" + chunk, config=cfg)
        return resp.text

    def extract(self, chunk):
        last_err = None
        for model in MODEL_CHAIN:
            if self.working_model and model != self.working_model:
                # stick to the proven model, but allow fallback list order
                pass
            retries = 0
            while True:
                try:
                    text = self._call(model, chunk)
                    self.working_model = model
                    self.model_used[model] = self.model_used.get(model, 0) + 1
                    return text, model
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    code = re.search(r"\b(400|401|403|404|429|5\d\d)\b", msg)
                    code = code.group(1) if code else "?"
                    if code == "404":
                        last_err = msg[:200]
                        break  # next model in chain
                    if code in ("429", "503") or "429" in msg or "503" in msg:
                        if retries >= MAX_RETRIES_429:
                            last_err = msg[:200]
                            break
                        wait = 2 ** retries * 4
                        print(f"    [{model}] rate-limited ({code}), sleep {wait}s (retry {retries+1})", flush=True)
                        time.sleep(wait)
                        retries += 1
                        continue
                    if code in ("400",):
                        last_err = msg[:300]
                        break
                    # unknown: brief wait then next model
                    print(f"    [{model}] error {code}: {msg[:200]}", flush=True)
                    last_err = msg[:200]
                    time.sleep(5)
                    break
        raise RuntimeError(f"all models failed, last: {last_err}")


def parse_triples(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return []
        data = json.loads(m.group(0))
    out = data.get("triples", []) if isinstance(data, dict) else []
    return out if isinstance(out, list) else []


def load_aliases():
    raw = json.loads(ALIASES_PATH.read_text())
    manual = raw.get("manual", [])
    auto = raw.get("auto", {})
    # (variant_norm, type) -> canonical ; plus typeless fallback variant_norm -> (canonical, type)
    typed, untyped = {}, {}
    for e in manual:
        for v in [e["canonical"]] + e.get("variants", []):
            typed.setdefault((norm(v), e["type"]), e["canonical"])
            untyped.setdefault(norm(v), (e["canonical"], e["type"]))
    for v, info in auto.items():
        typed.setdefault((norm(v), info["type"]), info["canonical"])
    return raw, typed, untyped


def canonicalize(name, ntype, typed, untyped, auto, merge_log):
    key = norm(name)
    if (key, ntype) in typed:
        canon = typed[(key, ntype)]
        if canon != name.strip():
            merge_log.append({"variant": name.strip(), "canonical": canon,
                              "type": ntype, "via": "alias"})
        return canon
    if key in untyped:
        canon, etype = untyped[key]
        if etype == ntype:
            if canon != name.strip():
                merge_log.append({"variant": name.strip(), "canonical": canon,
                                  "type": ntype, "via": "alias"})
            return canon
        # same string, different type -> intentionally distinct (e.g. Bauhaus)
        return name.strip()
    # auto casefold merge: register exact-spelling variants under first-seen form
    return name.strip()


def validate(t):
    """Return (ok, reason). Enforces schema, direction, blocklists."""
    s, st, r, o, ot = (t.get("subject", "").strip(), t.get("subject_type", "").strip(),
                       t.get("relation", "").strip(), t.get("object", "").strip(),
                       t.get("object_type", "").strip())
    if not s or not o:
        return False, "empty-name"
    if st not in NODE_TYPES or ot not in NODE_TYPES:
        return False, "bad-node-type"
    if r not in RELATIONS:
        return False, "bad-relation"
    sub_ok, obj_ok = REL_TYPE_CONSTRAINTS[r]
    if st not in sub_ok or ot not in obj_ok:
        return False, "bad-direction"
    if norm(s) in GENERIC_BLOCK or norm(o) in GENERIC_BLOCK:
        return False, "generic-noun"
    if norm(s) in DROP_EXACT or norm(o) in DROP_EXACT:
        return False, "dropped-term"
    if (st == "Work" and norm(s) in WORK_GENERIC_EXACT) or \
       (ot == "Work" and norm(o) in WORK_GENERIC_EXACT):
        return False, "generic-work-exact"
    if (st == "Person" and norm(s) in NON_DESIGNER_BLOCK) or \
       (ot == "Person" and norm(o) in NON_DESIGNER_BLOCK):
        return False, "non-designer-person"
    if ot == "Work" and any(g in norm(o) for g in WORK_GENERIC_SUBSTR):
        return False, "generic-work"
    return True, ""


def main():
    limit_docs = int(os.environ.get("W2_LIMIT_DOCS", "0") or 0)
    only = sys.argv[1:]  # optional explicit filenames
    OUT.mkdir(exist_ok=True)
    docs = sorted(DOCS.glob("*.md"))
    if only:
        docs = [d for d in docs if d.name in only]
    if limit_docs:
        docs = docs[:limit_docs]

    done = set()
    if PROGRESS_PATH.exists():
        done = set(json.loads(PROGRESS_PATH.read_text()).get("done_chunks", []))
    existing = 0
    if TRIPLES_PATH.exists():
        with open(TRIPLES_PATH) as f:
            for _ in f:
                existing += 1

    ex = Extractor()
    n_new, n_raw, drop_reasons = 0, 0, {}
    with open(TRIPLES_PATH, "a") as fout:
        for di, doc in enumerate(docs):
            text = doc.read_text()
            chunks = chunk_text(text)
            for ci, ch in enumerate(chunks):
                cid = f"{doc.name}#{ci}"
                if cid in done:
                    continue
                try:
                    out_text, model = ex.extract(ch)
                except RuntimeError as e:
                    print(f"STOP at {cid}: {e}", flush=True)
                    json.dump({"done_chunks": sorted(done)}, open(PROGRESS_PATH, "w"))
                    print(f"progress saved ({len(done)} chunks). resume by re-running.", flush=True)
                    raise SystemExit(1)
                triples = parse_triples(out_text)
                n_raw += len(triples)
                for t in triples:
                    ok, reason = validate(t)
                    if not ok:
                        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                        continue
                    rec = {"subject": t["subject"].strip(),
                           "subject_type": t["subject_type"].strip(),
                           "relation": t["relation"].strip(),
                           "object": t["object"].strip(),
                           "object_type": t["object_type"].strip(),
                           "source_doc": doc.name,
                           "source_sentence": t.get("source_sentence", "").strip(),
                           "chunk_id": cid,
                           "evidence_ok": norm(t.get("source_sentence", ""))[:60] in norm(ch)
                           if t.get("source_sentence") else False}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_new += 1
                done.add(cid)
                json.dump({"done_chunks": sorted(done)}, open(PROGRESS_PATH, "w"))
                print(f"[{di+1}/{len(docs)}] {cid} ({len(ch)}ch): +{len(triples)} raw -> file", flush=True)
                time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"extract done: {n_raw} raw, {n_new} kept, drops={drop_reasons}")
    json.dump(ex.model_used, open(MODEL_USED_PATH, "w"), indent=2)
    print(f"models used: {ex.model_used}")
    build_graph()


def build_graph():
    import networkx as nx
    raw_aliases, typed, untyped = load_aliases()
    auto = dict(raw_aliases.get("auto", {}))
    merge_log, seen_triples = [], set()
    G = nx.MultiDiGraph()
    node_aliases = {}  # node_id -> set(variants)

    def add_node(name, ntype):
        canon = canonicalize(name, ntype, typed, untyped, auto, merge_log)
        nid = f"{ntype}:{canon.lower()}"
        if nid not in G:
            G.add_node(nid, label=canon, type=ntype)
            node_aliases[nid] = set()
        if name.strip() != canon:
            node_aliases[nid].add(name.strip())
        return nid

    n_edges, n_dup, n_post_drop = 0, 0, 0
    with open(TRIPLES_PATH) as f:
        for line in f:
            t = json.loads(line)
            if norm(t["subject"]) in DROP_EXACT or norm(t["object"]) in DROP_EXACT:
                n_post_drop += 1
                continue
            if (t["subject_type"] == "Work" and norm(t["subject"]) in WORK_GENERIC_EXACT) or \
               (t["object_type"] == "Work" and norm(t["object"]) in WORK_GENERIC_EXACT):
                n_post_drop += 1
                continue
            s = add_node(t["subject"], t["subject_type"])
            o = add_node(t["object"], t["object_type"])
            key = (s, t["relation"], o, t["source_doc"], t["source_sentence"])
            if key in seen_triples:
                n_dup += 1
                continue
            seen_triples.add(key)
            G.add_edge(s, o, key=f"{t['relation']}|{len(seen_triples)}",
                       relation=t["relation"],
                       source_doc=t["source_doc"],
                       source_sentence=t["source_sentence"])
            n_edges += 1

    for nid, variants in node_aliases.items():
        if variants:
            G.nodes[nid]["aliases"] = "; ".join(sorted(variants))
    nx.write_graphml(G, GRAPH_PATH)

    # persist auto-discovered casefold variants (first-seen spelling kept)
    raw_aliases["auto"] = auto
    json.dump(raw_aliases, open(ALIASES_PATH, "w"), ensure_ascii=False, indent=2)
    json.dump(merge_log, open(MERGE_LOG_PATH, "w"), ensure_ascii=False, indent=2)

    from collections import Counter
    type_dist = Counter(d["type"] for _, d in G.nodes(data=True))
    rel_dist = Counter(d["relation"] for _, _, d in G.edges(data=True))
    UG = G.to_undirected()
    comps = sorted(nx.connected_components(UG), key=len, reverse=True) if len(G) else []
    stats = {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "type_dist": dict(type_dist), "rel_dist": dict(rel_dist),
        "merge_count": len(merge_log),
        "dup_dropped": n_dup,
        "post_filter_dropped": n_post_drop,
        "n_components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "isolated": sum(1 for c in comps if len(c) == 1),
    }
    json.dump(stats, open(STATS_PATH, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {GRAPH_PATH}, {MERGE_LOG_PATH}")


if __name__ == "__main__":
    if "--graph-only" in sys.argv:
        build_graph()
    else:
        main()
