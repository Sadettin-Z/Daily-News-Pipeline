"""News analysis pipeline: comprehensive per-article briefings, then combined cluster+merge."""
import json
import os
import re
import threading
import time
from collections import Counter
from datetime import date

import requests
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "keys", ".env")
    _loaded = load_dotenv(_env_path, override=True, encoding="utf-8-sig")
    if _loaded:
        print(f"[env] loaded {_env_path}")
    else:
        print(f"[env] WARNING: load_dotenv could not read {_env_path} (missing/empty/wrong encoding)")
except ImportError:
    print("[env] python-dotenv not installed; using OS environment only")

from scraper import collect_articles, RSS_FEEDS, TURKISH_SOURCES, NON_ENGLISH_INTERNATIONAL_SOURCES, normalize_turkish_chars


# Relevance criteria for the post-summary filter. Edit freely; the LLM reads this verbatim.
# Bias is intentionally conservative — when unsure, the article is kept.
INTERESTS = """
You are filtering news for a reader who follows world affairs and Turkish politics/economy.
When genuinely unsure between keep and drop, KEEP.

KEEP:
- International geopolitics, diplomacy, wars, conflicts, peace processes
- Turkish politics, government decisions, judicial cases of national significance
- Turkish economy, lira, central bank, major Turkish corporate news
- Foreign policy decisions affecting Türkiye, the Middle East, or major powers
- Major global economic developments — markets, central banks, trade policy, sanctions, large M&A
- Significant technology / science breakthroughs with broad social, economic, or geopolitical implications
- Major business stories with industry-wide or regulatory impact (antitrust, large bankruptcies, major investigations)
- Investigative journalism, exposés, leaks of public interest
- Major social/political movements, protests, civil unrest
- Climate, energy, environment policy and major related events
- Health policy, pandemics, treatment or vaccine breakthroughs, major medical or public-health developments — not single observational studies
- Major attacks, large-scale violence with political dimension
- Actions or statements by globally recognized heads of state (Trump, Netanyahu, Putin, Xi, Macron, Modi, etc.) that reveal strategy or political project
- Country-internal stories carrying a signal beyond their borders: precedent-setting rulings on issues of broad international debate (rights, speech, antitrust, religion), institutional failures revealing capacity problems, patterns of religious/ethnic violence, direct cross-border consequences (contagion, refugee flow, conflict spread)

DROP:
- Sports of any kind — football, basketball, NBA, tennis, F1, Olympics, individual athletes, match results, transfers
- Celebrity gossip, entertainment industry, TV shows, awards, personal lives of celebrities
- Lifestyle content — DIY tips, recipes, cleaning hacks, "amazing trick" listicles, before/after transformations
- Hyper-local incidents — single fires, single accidents, individual crimes — unless they have broader political/policy implications
- Minor business stories — small-company layoffs, single-store closures, small class actions, gadget releases, app updates
- Routine product reviews, unboxings, "best of" lists
- Weather — unless major disaster (hurricane landfall, large earthquake)
- Clickbait — "you won't believe", "amazing", "one weird trick"
- Routine sub-national politics — state redistricting, congressional primary endorsements, local party realignments, state-legislator behavior
- Personnel or institutional announcements — CEO replacements, news-director appointments, board reshuffles, ambassador appointments without scandal
- Stories centered on figures the international reader wouldn't recognize (state-level officials, mid-tier executives, sub-national party figures) unless they stand in for a larger trend
- Quirky crime, fraud, or eccentric corruption cases — entertaining is not significant
- Procedural court steps (filing made, motion submitted, trial begins, hearing scheduled) where nothing unusual has happened yet
- Local commercial or environmental disputes without national or industry-wide impact
- Single accidents (road, rail, air) without political or terror dimension, even with high casualties
- Domestic stories with no cross-border signal, no globally recognized figure, and no precedent-setting outcome
""".strip()


# Up to 4 API keys read from env: GEMINI_API_KEY_1 ... GEMINI_API_KEY_4
# Falls back to GEMINI_API_KEY if numbered keys are not set.
PRIMARY_MODEL = "gemini-3.6-flash"

_raw_keys = {f"GEMINI_API_KEY_{i}": os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 5)}
GEMINI_KEYS = [v for v in _raw_keys.values() if v]
if not GEMINI_KEYS:
    fallback = os.environ.get("GEMINI_API_KEY")
    if fallback:
        GEMINI_KEYS = [fallback]
print(f"Loaded {len(GEMINI_KEYS)} Gemini API key(s): " +
      (", ".join(name for name, val in _raw_keys.items() if val) or "(fallback GEMINI_API_KEY)"))


_key_lock = threading.Lock()
_key_counter = 0


def _rotated_keys():
    """Return [(key_idx_1based, key), ...] rotated so a different key starts each call."""
    global _key_counter
    if not GEMINI_KEYS:
        return []
    with _key_lock:
        start = _key_counter % len(GEMINI_KEYS)
        _key_counter += 1
    n = len(GEMINI_KEYS)
    return [(((start + i) % n) + 1, GEMINI_KEYS[(start + i) % n]) for i in range(n)]


def _short_err(e):
    """Status header + message from a google-genai exception. Drops JSON details and boilerplate URLs."""
    raw = str(e)
    head = raw.split("{", 1)[0].strip().rstrip(".")
    bits = [head] if head else []
    m = re.search(
        r"'message':\s*'(.*?)'\s*,\s*'(?:status|code|details|@type)'",
        raw, re.DOTALL,
    )
    if m:
        msg = m.group(1).replace("\\n", " ")
        msg = re.sub(r"For more information on this error[^.]*\.\s*", "", msg)
        msg = re.sub(r"To monitor your current usage[^.]*\.\s*", "", msg)
        msg = re.sub(r"\s+", " ", msg).strip(" *")
        if len(msg) > 250:
            msg = msg[:250] + "..."
        if msg:
            bits.append(msg)
    return " — ".join(bits) if bits else type(e).__name__


def _build_config(model, response_schema):
    thinking = types.ThinkingConfig(thinking_level="high")
    if response_schema is not None:
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=thinking,
        )
    return types.GenerateContentConfig(thinking_config=thinking)


def gemini_generate(contents, response_schema=None):
    """Run PRIMARY_MODEL across all keys, retrying transient failures."""
    if not GEMINI_KEYS:
        raise RuntimeError("No Gemini API keys configured. Set GEMINI_API_KEY_1..4 or GEMINI_API_KEY.")

    quota_signals = ("429", "quota", "rate limit", "rate_limit", "resource_exhausted", "exhausted")
    transient_signals = ("503", "unavailable", "overloaded", "high demand", "502", "504", "499", "deadline", "timeout", "timed out", "cancelled", "canceled", "empty response", "server disconnected", "remoteprotocolerror", "connection reset", "connection aborted", "connection closed", "broken pipe", "incomplete read")
    not_found_signals = ("404", "not_found", "not found", "is not supported")

    last_err = None
    last_was_transient = False
    model = PRIMARY_MODEL
    config = _build_config(model, response_schema)

    MAX_MODEL_ATTEMPTS = 3       # initial pass + up to 2 retries on all-transient rounds
    TRANSIENT_RETRY_WAIT = 20    # base seconds; multiplied by attempt number for backoff

    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        round_quota = 0
        round_transient = 0

        for key_idx, key in _rotated_keys():
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=180_000),
            )
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                if not response.text:
                    finish, block = "?", "?"
                    try:
                        if response.candidates:
                            finish = str(response.candidates[0].finish_reason)
                        if response.prompt_feedback and response.prompt_feedback.block_reason:
                            block = str(response.prompt_feedback.block_reason)
                    except Exception:
                        pass
                    raise RuntimeError(f"empty response from model (finish={finish}, block={block})")
                return response
            except Exception as e:
                msg = str(e).lower()
                last_err = e
                if any(s in msg for s in not_found_signals):
                    raise RuntimeError(f"Model '{model}' not available on these keys: {_short_err(e)}")
                if any(s in msg for s in quota_signals):
                    print(f"  [{model} | key {key_idx}/{len(GEMINI_KEYS)}] quota/rate-limit, rotating: {_short_err(e)}")
                    round_quota += 1
                    last_was_transient = False
                    continue
                if any(s in msg for s in transient_signals):
                    print(f"  [{model} | key {key_idx}/{len(GEMINI_KEYS)}] transient error, rotating: {_short_err(e)}")
                    round_transient += 1
                    last_was_transient = True
                    continue
                print(f"  [{model} | key {key_idx}/{len(GEMINI_KEYS)}] non-retryable error: {_short_err(e)}")
                raise

        # All keys exhausted this round. Retry if at least one failure was transient.
        if round_transient > 0 and attempt < MAX_MODEL_ATTEMPTS:
            wait = TRANSIENT_RETRY_WAIT * attempt
            print(f"  '{model}' round {attempt}/{MAX_MODEL_ATTEMPTS} done (quota={round_quota}, transient={round_transient}). Waiting {wait}s before retrying transient keys...")
            time.sleep(wait)
            continue
        break  # all keys quota-dead — give up

    if last_was_transient:
        raise RuntimeError(
            f"'{PRIMARY_MODEL}' unavailable on Google's side (503/overload) across all "
            f"{len(GEMINI_KEYS)} key(s). Wait a few minutes and retry. Last error: {last_err}"
        )
    raise RuntimeError(
        f"All Gemini keys exhausted on '{PRIMARY_MODEL}'. Last error: {last_err}"
    )


def _normalize_strings(obj):
    """Walk a parsed JSON structure and normalize every string value."""
    if isinstance(obj, str):
        return normalize_turkish_chars(obj)
    if isinstance(obj, list):
        return [_normalize_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_strings(v) for k, v in obj.items()}
    return obj


def parse_json_lenient(text):
    """Parse Gemini JSON output, repairing malformed \\uXXXX escapes, then normalize
    every string in the parsed result. Post-parse normalization is required because
    Gemini may emit a corrupted char as a \\uXXXX escape (e.g. \\u0159 for ř) — those
    only become real characters after json.loads, so a raw-text pass would miss them."""
    # First, normalize the raw JSON in case literal corrupted chars sit inside it.
    text = normalize_turkish_chars(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Gemini occasionally emits a stray \u not followed by 4 hex digits.
        # Re-escape those backslashes so the literal characters survive parsing.
        fixed = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)
        data = json.loads(fixed)
    return _normalize_strings(data)


SUMMARIZE_BATCH_SIZE = 10  # Max articles per Gemini call. More than this hurts per-article quality.


def summarize_source(source_name, articles):
    """Stage 1: summarize articles from one source. Splits into batches of SUMMARIZE_BATCH_SIZE
    (e.g. 23 articles → batches of 10, 10, 3) and makes one Gemini call per batch."""
    if not articles:
        return []

    batches = [
        articles[i:i + SUMMARIZE_BATCH_SIZE]
        for i in range(0, len(articles), SUMMARIZE_BATCH_SIZE)
    ]

    out = []
    for batch_idx, batch in enumerate(batches, 1):
        if len(batches) > 1:
            print(f"  Batch {batch_idx}/{len(batches)} ({len(batch)} articles)")
        try:
            out.extend(_summarize_batch(source_name, batch))
        except Exception as e:
            print(f"  ! [{source_name}] batch {batch_idx}/{len(batches)} failed, "
                  f"skipping its {len(batch)} article(s): {e}")
    return out


def _summarize_batch(source_name, articles):
    """Send one batch (≤ SUMMARIZE_BATCH_SIZE articles) in a single Gemini call."""
    translate_titles = source_name in NON_ENGLISH_INTERNATIONAL_SOURCES

    payload = "\n\n---\n\n".join(
        f"[{i}] TITLE: {a['title']}\nURL: {a['url']}\nTEXT: {a['text']}"
        for i, a in enumerate(articles)
    )

    item_props = {
        "index": {"type": "integer"},
        "summary": {"type": "string"},
    }
    required = ["index", "summary"]
    if translate_titles:
        item_props["title_en"] = {"type": "string"}
        required.append("title_en")

    schema = {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": required,
                },
            },
        },
        "required": ["summaries"],
    }

    title_instruction = (
        "  - Also provide title_en: a faithful English translation of the article's original title. "
        "Preserve names, places, numbers, and quoted phrases verbatim. Do not editorialize.\n"
        if translate_titles else ""
    )

    prompt = (
        f"Articles:\n\n{payload}\n\n"

        "Based on the articles above, produce one clear news briefing per article. "
        "Language rule: Turkish source → write the briefing in Turkish. "
        "Any other source → write the briefing in English (translate if the source is in another language).\n\n"

        "For each numbered article, follow this procedure internally before writing:\n"
        "  STEP 1 — Identify the facts: the people (full names + roles), the organizations and places, "
        "the events, the numbers and dates, the causes and consequences, and the direct quotes "
        "that add substance.\n"
        "  STEP 2 — Draft a tight briefing covering these facts in flowing prose.\n"
        "  STEP 3 — Tighten the WORDING, not the information: cut journalistic boilerplate "
        "(grant funders, editors, photo credits), rhetorical filler, and quotes that restate "
        "what another quote already said. Keep every concrete fact — names, numbers, dates, "
        "causes, consequences, and the context needed to understand why the event matters. "
        "Brevity must come from tighter sentences, not from dropping facts.\n\n"

        "Constraints:\n"
        "  - Stay tight. Only stretch when an article genuinely covers multiple distinct events.\n"
        "  - Preserve central numbers, dates, full names, and titles VERBATIM.\n"
        "  - Preserve the cause-effect chain and sequence of events.\n"
        "  - Flowing prose. No bullet points, no headings, no markdown.\n"
        "  - Do NOT add information that is not in the source. Do NOT speculate or infer.\n"
        f"{title_instruction}"
        "  - Return one entry per article using the article's index."
    )

    response = gemini_generate(prompt, response_schema=schema)
    data = parse_json_lenient(response.text)

    out = []
    seen_indices = set()
    bad_indices = []
    for entry in data.get("summaries", []):
        i = entry.get("index", -1)
        if not (0 <= i < len(articles)):
            bad_indices.append(i)
            continue
        if i in seen_indices:
            continue
        seen_indices.add(i)
        title = articles[i]["title"]
        if translate_titles:
            title_en = (entry.get("title_en") or "").strip()
            if title_en:
                title = title_en
        out.append({
            "source": source_name,
            "title": title,
            "url": articles[i]["url"],
            "summary": entry["summary"],
        })
    missing = [i for i in range(len(articles)) if i not in seen_indices]
    if missing or bad_indices:
        print(f"  ! [{source_name}] batch sonucu: {len(out)}/{len(articles)} özet alındı"
              + (f", eksik index'ler: {missing}" if missing else "")
              + (f", geçersiz index'ler: {bad_indices}" if bad_indices else ""))
    return out


def cluster_summaries(summaries):
    """Stage 2: group summaries that report the same news event. Returns list of index lists."""
    payload = "\n\n".join(
        f"[{i}] ({s['source']}) {s['title']}\n{s['summary']}"
        for i, s in enumerate(summaries)
    )

    schema = {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["indices"],
                },
            },
        },
        "required": ["clusters"],
    }

    prompt = (
        f"News items:\n\n{payload}\n\n"
        "Group the items above so each group covers the same underlying news event. "
        "Items about the same story (even from different sources or in different languages) belong together. "
        "Items unique to one event form a single-item group. "
        "Every index must appear in exactly ONE group. No duplicates, no omissions."
    )

    try:
        response = gemini_generate(prompt, response_schema=schema)
        data = parse_json_lenient(response.text)
        return [c["indices"] for c in data.get("clusters", [])]
    except Exception as e:
        print(f"  ! Clustering failed ({e}); treating every summary as its own story.")
        return [[i] for i in range(len(summaries))]


def merge_cluster(items):
    """Stage 3: merge multi-source items into (title, summary). Singletons skip the LLM and reuse the source's own title."""
    if len(items) == 1:
        return items[0]["title"], items[0]["summary"]

    payload = "\n\n".join(
        f"({it['source']}) {it['title']}\n{it['summary']}"
        for it in items
    )

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["title", "summary"],
    }

    prompt = (
        f"Items:\n\n{payload}\n\n"

        "The items above report the SAME news event from different sources. "
        "Based on them, produce TWO outputs in one response:\n"
        "  1. title — a headline (≤15 words) that pulls the reader in while staying honest. "
        "Be specific: concrete actor + concrete action + the story's most striking element "
        "(unexpected outcome, contradiction, number, confrontation), taken from the sources. "
        "Active verb, key fact front-loaded. Curiosity is fine but the title's promise must be "
        "answered in the summary: no withholding phrases ('what happened next', 'shocking decision'), "
        "no ALL-CAPS or exclamations, no unanswered question headlines, no hype adjectives — "
        "the striking element must come from the story's own facts.\n"
        "  2. summary — ONE unified briefing that merges the items into a single readable account.\n\n"

        "How to merge:\n"
        "  - The items overlap heavily. State each shared fact ONCE — never stack near-identical "
        "tellings of the same fact from different sources.\n"
        "  - Add the substantive facts that only some sources mention: names, numbers, dates, "
        "causes, consequences, key quotes.\n"
        "  - Drop non-informative color: reaction quotes that add no new fact, scene-setting, "
        "procedural minutiae. Substance stays, filler goes.\n"
        "  - If sources disagree on a fact, give both versions with attribution "
        "(e.g., 'Reuters reports 50 dead while AP reports 47').\n"
        "  - Open with whatever matters most in THIS story — judge per story; sometimes it is a person, "
        "sometimes a number, sometimes a decision — then continue in decreasing importance.\n\n"

        "Constraints:\n"
        "  - Preserve numbers, dates, full names, titles, and direct quotes VERBATIM.\n"
        "  - Preserve the cause-effect chain.\n"
        "  - Flowing prose. No bullets, no headings, no markdown.\n"
        "  - Language for BOTH title and summary: use the language of the MAJORITY of items; if tied, use Turkish.\n"
        "  - Do NOT invent details. Do NOT speculate. Length must come from the sources' actual content."
    )

    response = gemini_generate(prompt, response_schema=schema)
    data = parse_json_lenient(response.text)
    return data["title"], data["summary"]


def filter_summaries(summaries):
    """Drop summaries that don't match INTERESTS. Falls back to keeping all on filter failure
    so a single bad LLM response can't wipe out the report."""
    if not summaries:
        return summaries

    payload = "\n\n".join(
        f"[{i}] ({s['source']}) {s['title']}\n{s['summary']}"
        for i, s in enumerate(summaries)
    )

    schema = {
        "type": "object",
        "properties": {
            "drop_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["drop_indices"],
    }

    prompt = (
        f"Items (each with index, source, title, briefing):\n\n{payload}\n\n"
        f"{INTERESTS}\n\n"
        "Apply the criteria above to the items. Return drop_indices: the indices of items to DROP. "
        "When genuinely unsure, KEEP (omit the index). Return only drop_indices."
    )

    print(f"Filtering {len(summaries)} summaries against interest criteria...")
    try:
        response = gemini_generate(prompt, response_schema=schema)
        data = parse_json_lenient(response.text)
        drop = {i for i in data.get("drop_indices", []) if 0 <= i < len(summaries)}
    except Exception as e:
        print(f"  Filter call failed ({e}); keeping all.")
        return summaries

    kept = [s for i, s in enumerate(summaries) if i not in drop]
    print(f"  Kept {len(kept)}/{len(summaries)} ({len(drop)} dropped)")
    if drop:
        print("  Dropped:")
        for i in sorted(drop):
            print(f"    - [{summaries[i]['source']}] {summaries[i]['title']}")
    return kept


def main():
    articles_by_source = collect_articles()

    all_summaries = []
    for source, articles in articles_by_source.items():
        print(f"Summarizing {len(articles)} articles from {source}...")
        all_summaries.extend(summarize_source(source, articles))

    if not all_summaries:
        print("No articles collected.")
        return

    all_summaries = filter_summaries(all_summaries)
    if not all_summaries:
        print("Interest filter dropped everything — nothing to report.")
        return

    print(f"Clustering {len(all_summaries)} summaries...")
    clusters = cluster_summaries(all_summaries)

    seen = set()
    duplicate_count = 0
    cluster_items = []
    for cluster_indices in clusters:
        cleaned = []
        for i in cluster_indices:
            if not (0 <= i < len(all_summaries)):
                continue
            if i in seen:
                duplicate_count += 1
                continue
            seen.add(i)
            cleaned.append(all_summaries[i])
        if cleaned:
            cluster_items.append(cleaned)

    missing = [i for i in range(len(all_summaries)) if i not in seen]
    if missing:
        print(f"  ! {len(missing)} özet hiçbir cluster'a girmedi, singleton olarak ekleniyor: {missing}")
        for i in missing:
            cluster_items.append([all_summaries[i]])
    if duplicate_count:
        print(f"  ! {duplicate_count} özet birden fazla cluster'da geçti, ilki tutuldu")

    size_counts = Counter(len(items) for items in cluster_items)
    multi = sum(c for size, c in size_counts.items() if size > 1)
    singletons = size_counts.get(1, 0)
    dist = ", ".join(f"{c} cluster(s) with {size} sources" for size, c in sorted(size_counts.items()))
    print(f"Cluster size distribution: {dist}")
    print(f"Merging {len(cluster_items)} clusters serially ({multi} need an API call, {singletons} singletons skip)...")

    output = []
    merge_idx = 0
    for items in cluster_items:
        if len(items) > 1:
            merge_idx += 1
            sources_list = ", ".join(it["source"] for it in items)
            print(f"  [merge {merge_idx}/{multi}] {len(items)} sources: {sources_list}")
        try:
            title, merged = merge_cluster(items)
        except Exception as e:
            print(f"  ! Merge failed ({e}); stacking the source summaries unmerged.")
            title = items[0]["title"]
            merged = "\n\n".join(f"({it['source']}) {it['summary']}" for it in items)
        output.append({
            "title": title,
            "summary": merged,
            "sources": [
                {"source": it["source"], "title": it["title"], "url": it["url"]}
                for it in items
            ],
        })

    with open("news_clusters.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output)} clusters to news_clusters.json")

    messages = build_discord_messages(output)

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook:
        print(f"Posting {len(messages)} chunked messages to Discord webhook...")
        send_to_discord(messages, webhook)
        print("Pausing 10s before file upload to let Discord rate-limit window drain...")
        time.sleep(10)
        parts = []
        for fname, text in build_report_files(output):
            plain = strip_markdown(text)
            if len(plain.encode("utf-8")) > 48_000:
                chunks = split_report(plain)
                n = len(chunks)
                parts.extend((f"{fname[:-3]} {i+1}_of_{n}.md", c) for i, c in enumerate(chunks))
            else:
                parts.append((fname, plain))
        upload_report_to_discord(
            parts,
            webhook,
            f"**Haber Kaynakları {date.today().strftime('%d %m %Y')}**",
        )
    else:
        print("DISCORD_WEBHOOK_URL not set in env — skipping Discord. Add it to keys/.env to enable.")


def _is_turkish_cluster(entry):
    """A cluster counts as Turkish if more than half its sources are Turkish-language feeds.
    Used to sort multi-source stories: English clusters first, Turkish clusters last."""
    n_tr = sum(1 for s in entry["sources"] if s["source"] in TURKISH_SOURCES)
    return n_tr * 2 > len(entry["sources"])


def _safe_filename(name, max_len=100):
    """Strip characters Discord/Windows reject and cap the length."""
    name = re.sub(r"[\\/]", "-", name)
    name = re.sub(r'[:*?"<>|\n\r\t]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:max_len].rstrip(" .") or "Rapor"


def build_report_files(output):
    """One (filename, text) part per multi-source story — filename is the title plus the
    covering sources — then one part per source holding its single-source stories."""
    multi = [e for e in output if len(e["sources"]) > 1]
    # Most-covered stories first; ties keep English clusters before Turkish ones.
    multi.sort(key=lambda e: (-len(e["sources"]), _is_turkish_cluster(e)))
    singles = [e for e in output if len(e["sources"]) == 1]

    by_source = {}
    for entry in singles:
        by_source.setdefault(entry["sources"][0]["source"], []).append(entry)

    files = []
    used = set()

    def add_file(base, text):
        fname = _safe_filename(base)
        n = 2
        while fname.lower() in used:
            fname = _safe_filename(base, max_len=95) + f" ({n})"
            n += 1
        used.add(fname.lower())
        files.append((fname + ".md", text))

    for entry in multi:
        src_names = [name for name, _ in Counter(s["source"] for s in entry["sources"]).most_common()]
        lines = [f"## {entry['title']}", "", entry["summary"], "", "**Sources:**"]
        lines.extend(f"- [{s['source']} — {s['title']}]({s['url']})" for s in entry["sources"])
        add_file(f"{entry['title']} - {', '.join(src_names)}", "\n".join(lines))

    for src in RSS_FEEDS:
        if src not in by_source:
            continue
        lines = [f"# {src}", ""]
        for entry in by_source[src]:
            lines += [f"## {entry['title']}", "", entry["summary"], "",
                      f"[Read more]({entry['sources'][0]['url']})", "", "---", ""]
        add_file(src, "\n".join(lines))

    return files


DISCORD_LIMIT = 1990
DISCORD_SUPPRESS_EMBEDS = 4


def chunk_for_discord(text, limit=DISCORD_LIMIT):
    """Split text into ≤ limit-char pieces. Never cuts mid-word unless a single token exceeds the limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = None
        for sep in ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "):
            pos = window.rfind(sep)
            if pos >= limit // 2:
                cut = pos + len(sep)
                break
        if cut is None:
            pos = window.rfind(" ")
            if pos > 0:
                cut = pos + 1
            else:
                cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _article_to_payloads(entry, index_prefix=""):
    title = index_prefix + entry["title"]
    summary = entry["summary"]
    sources = entry["sources"]
    primary_url = sources[0]["url"]
    primary_source = sources[0]["source"]

    if len(sources) > 1:
        # A cluster may hold several articles from the same source — show each source once,
        # with its article count when > 1 (e.g. "bbc.com 3 · aljazeera.com").
        counts = Counter(s["source"] for s in sources)
        tail = " · ".join(f"{src} {n}" if n > 1 else src for src, n in counts.most_common())
        header = f"## [{title}](<{primary_url}>)\n-# Covered by: {tail}\n\n"
    else:
        header = f"## [{title}](<{primary_url}>)\n-# {primary_source}\n\n"

    full = header + summary
    payloads = [
        {"content": c, "flags": DISCORD_SUPPRESS_EMBEDS}
        for c in chunk_for_discord(full, limit=DISCORD_LIMIT)
    ]

    if len(sources) > 1:
        links = "\n".join(f"• [{s['source']} — {s['title'][:80]}](<{s['url']}>)" for s in sources)
        link_msg = f"**Sources:**\n{links}"
        payloads.extend(
            {"content": c, "flags": DISCORD_SUPPRESS_EMBEDS}
            for c in chunk_for_discord(link_msg, limit=DISCORD_LIMIT)
        )
    return payloads


def _section_header(title, subtitle=None):
    content = f"# {title}"
    if subtitle:
        content += f"\n-# {subtitle}"
    return {"content": content, "flags": DISCORD_SUPPRESS_EMBEDS}


def build_discord_messages(output):
    """Build the list of webhook payload dicts. Ping is attached to the first payload only."""
    multi = [e for e in output if len(e["sources"]) > 1]
    # Most-covered stories first; ties keep English clusters before Turkish ones.
    multi.sort(key=lambda e: (-len(e["sources"]), _is_turkish_cluster(e)))
    singles = [e for e in output if len(e["sources"]) == 1]

    by_source = {}
    for entry in singles:
        src = entry["sources"][0]["source"]
        by_source.setdefault(src, []).append(entry)

    first = _section_header(f"📰 News Report — {date.today().isoformat()}",
                            subtitle=f"{len(multi)} cross-source stories · {len(singles)} unique stories")
    first["content"] = f"<@331085172264927235>\n{first['content']}"
    first["allowed_mentions"] = {"parse": [], "users": ["331085172264927235"]}
    payloads = [first]

    if multi:
        payloads.append(_section_header("🔥 Common News",
                                        subtitle="Stories covered by multiple sources"))
        for i, entry in enumerate(multi, 1):
            payloads.extend(_article_to_payloads(entry, index_prefix=f"{i}. "))

    for src in RSS_FEEDS:
        if src not in by_source:
            continue
        payloads.append(_section_header(f"📡 {src}",
                                        subtitle=f"{len(by_source[src])} stories"))
        for entry in by_source[src]:
            payloads.extend(_article_to_payloads(entry))

    return payloads


def send_to_discord(payloads, webhook_url, delay=2.0):
    """POST each chunked payload to the webhook. Respects 429 retry_after."""
    total = len(payloads)
    start = time.time()
    for i, payload_dict in enumerate(payloads, 1):
        attempts = 0
        while True:
            attempts += 1
            try:
                r = requests.post(webhook_url, json=payload_dict, timeout=30)
                if r.status_code == 429 and attempts < 3:
                    try:
                        retry = r.json().get("retry_after", 5)
                    except Exception:
                        retry = 5
                    print(f"  [discord {i}/{total}] rate-limited, sleeping {retry}s", flush=True)
                    time.sleep(float(retry) + 0.5)
                    continue
                if r.status_code >= 400:
                    print(f"  [discord {i}/{total}] HTTP {r.status_code}: {r.text[:200]}", flush=True)
                break
            except Exception as e:
                print(f"  [discord {i}/{total}] error: {e}", flush=True)
                break

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  [discord progress] {i}/{total} sent ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)

        if i < total:
            time.sleep(delay)


def strip_markdown(text):
    """Convert markdown to decorated plain text: underlined headings, UPPERCASE labels, • bullets."""
    text = re.sub(r'^\*\s+', '- ', text, flags=re.MULTILINE)

    def heading_repl(m):
        title = m.group(1).strip()
        underline = "─" * min(len(title), 60)
        return f"\n{title}\n{underline}\n"

    text = re.sub(r'^#{1,6}\s+(.+)$', heading_repl, text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\(<?([^)>]+)>?\)', r'\1', text)
    text = re.sub(r'^Sources:', 'SOURCES:', text, flags=re.MULTILINE)
    text = re.sub(r'^- ', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^(?!• )(.+)\n(?=• )', r'\1\n\n', text, flags=re.MULTILINE)
    text = re.sub(r'(^• .+)\n(?=• )', r'\1\n\n', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def split_report(text, max_bytes=48_000):
    """Split report text into parts <= max_bytes (UTF-8), breaking only at --- separators between stories.
    Discord's in-browser file viewer truncates around ~50 KB even when 'View whole document' is clicked,
    so we cap below that to keep every part fully previewable."""
    parts = re.split(r'(?m)^---\s*$\n?', text)
    blocks = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            blocks.append(p + '---\n')
        elif p.strip():
            blocks.append(p)

    files = []
    current = ""
    for block in blocks:
        if current and len((current + block).encode("utf-8")) > max_bytes:
            files.append(current)
            current = block
        else:
            current += block
    if current:
        files.append(current)
    return files


def upload_report_to_discord(parts, webhook_url, message):
    """Upload (filename, text) parts as .md attachments. No ping (already fired).
    Discord allows max 10 files per message, so parts go out in batches; the
    message text rides on the first batch only."""
    total_chars = sum(len(text) for _, text in parts)
    print(f"Uploading {len(parts)} file(s) ({total_chars:,} chars total) to Discord...")
    for batch_no, start in enumerate(range(0, len(parts), 10), 1):
        batch = parts[start:start + 10]
        content = message if start == 0 else f"{message} (devam {batch_no})"
        _upload_file_batch(batch, webhook_url, content)
        if start + 10 < len(parts):
            time.sleep(2)


def _upload_file_batch(batch, webhook_url, message):
    payload = {"content": message}

    for attempt in range(1, 6):
        files = [("payload_json", (None, json.dumps(payload), "application/json"))]
        for i, (fname, text) in enumerate(batch):
            files.append((f"files[{i}]", (fname, text.encode("utf-8"), "text/markdown")))
        try:
            r = requests.post(webhook_url, files=files, timeout=60)
            if r.status_code == 429 and attempt < 5:
                try:
                    retry = r.json().get("retry_after", 5)
                except Exception:
                    retry = 5
                wait = float(retry) + 1.0
                print(f"  Discord upload rate-limited (attempt {attempt}/5), sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                print(f"  Discord HTTP {r.status_code}: {r.text[:300]}")
            else:
                print(f"  Discord upload OK")
            return
        except Exception as e:
            print(f"  Discord upload error (attempt {attempt}/5): {e}")
            if attempt < 5:
                time.sleep(5)
    print(f"  Discord upload FAILED after 5 attempts")


if __name__ == "__main__":
    main()
