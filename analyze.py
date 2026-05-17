"""News analysis pipeline: comprehensive per-article briefings, then combined cluster+merge."""
import json
import os
import re
import threading
import time

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

from scraper import collect_articles, RSS_FEEDS


# Up to 4 API keys read from env: GEMINI_API_KEY_1 ... GEMINI_API_KEY_4
# Falls back to GEMINI_API_KEY if numbered keys are not set.
PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-2.5-pro"
FALLBACK_THINKING_BUDGET = 24576  # Gemini API max for 2.5-pro; 2.5 series uses thinking_budget (int), not thinking_level

_raw_keys = {f"GEMINI_API_KEY_{i}": os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 5)}
GEMINI_KEYS = [v for v in _raw_keys.values() if v]
if not GEMINI_KEYS:
    fallback = os.environ.get("GEMINI_API_KEY")
    if fallback:
        GEMINI_KEYS = [fallback]
print(f"Loaded {len(GEMINI_KEYS)} Gemini API key(s): " +
      ", ".join(name for name, val in _raw_keys.items() if val) or "(fallback GEMINI_API_KEY)")


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
    # Gemini 3+ uses thinking_level; 2.5 series uses thinking_budget (mixing the two returns 400).
    if model.startswith("gemini-2.5"):
        thinking = types.ThinkingConfig(thinking_budget=FALLBACK_THINKING_BUDGET)
    else:
        thinking = types.ThinkingConfig(thinking_level="high")
    if response_schema is not None:
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=thinking,
        )
    return types.GenerateContentConfig(thinking_config=thinking)


def gemini_generate(contents, response_schema=None):
    """Try PRIMARY_MODEL across all keys, then fall back to FALLBACK_MODEL across all keys."""
    if not GEMINI_KEYS:
        raise RuntimeError("No Gemini API keys configured. Set GEMINI_API_KEY_1..4 or GEMINI_API_KEY.")

    quota_signals = ("429", "quota", "rate limit", "rate_limit", "resource_exhausted", "exhausted")
    transient_signals = ("503", "unavailable", "overloaded", "high demand", "502", "504", "499", "deadline", "timeout", "timed out", "cancelled", "canceled", "empty response")
    not_found_signals = ("404", "not_found", "not found", "is not supported")

    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_err = None
    last_was_transient = False

    MAX_MODEL_ATTEMPTS = 3       # initial pass + up to 2 retries on all-transient rounds
    TRANSIENT_RETRY_WAIT = 20    # base seconds; multiplied by attempt number for backoff

    for i, model in enumerate(models):
        next_model = models[i + 1] if i + 1 < len(models) else None
        config = _build_config(model, response_schema)
        model_unavailable = False

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
                        print(f"  [{model}] not available on these keys: {_short_err(e)}")
                        model_unavailable = True
                        break  # 404 is the same on every key — go straight to next model
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

            if model_unavailable:
                break  # skip retries, go to next model

            # All keys exhausted this round. Retry the same model if at least one failure was transient
            # (those keys may recover; quota'd keys will fail again instantly but cost ~1s each).
            # Skip retry only if EVERY key is quota-dead — no possibility of recovery on this model today.
            if round_transient > 0 and attempt < MAX_MODEL_ATTEMPTS:
                wait = TRANSIENT_RETRY_WAIT * attempt
                print(f"  '{model}' round {attempt}/{MAX_MODEL_ATTEMPTS} done (quota={round_quota}, transient={round_transient}). Waiting {wait}s before retrying transient keys...")
                time.sleep(wait)
                continue
            break  # all keys quota-dead on this model — fall through to next model

        if next_model:
            print(f"  Falling back from '{model}' to '{next_model}'...")

    if last_was_transient:
        raise RuntimeError(
            f"Both '{PRIMARY_MODEL}' and '{FALLBACK_MODEL}' are unavailable on Google's side "
            f"(503/overload) across all {len(GEMINI_KEYS)} key(s). Wait a few minutes and retry. "
            f"Last error: {last_err}"
        )
    raise RuntimeError(
        f"All Gemini keys exhausted across primary '{PRIMARY_MODEL}' and fallback '{FALLBACK_MODEL}'. "
        f"Last error: {last_err}"
    )


def parse_json_lenient(text):
    """Parse Gemini JSON output, repairing malformed \\uXXXX escapes if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Gemini occasionally emits a stray \u not followed by 4 hex digits.
        # Re-escape those backslashes so the literal characters survive parsing.
        fixed = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)
        return json.loads(fixed)


def summarize_source(source_name, articles):
    """Stage 1: batch-summarize all articles from one source. Returns list of dicts."""
    if not articles:
        return []

    payload = "\n\n---\n\n".join(
        f"[{i}] TITLE: {a['title']}\nURL: {a['url']}\nTEXT: {a['text']}"
        for i, a in enumerate(articles)
    )

    schema = {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "summary": {"type": "string"},
                    },
                    "required": ["index", "summary"],
                },
            },
        },
        "required": ["summaries"],
    }

    prompt = (
        "You are producing COMPREHENSIVE, detail-preserving briefings of news articles. "
        "These are NOT short abstracts. Coverage of every salient fact beats brevity. "
        "Write each briefing in the same language as the source article "
        "(Turkish stays Turkish, English stays English).\n\n"

        "For each numbered article, follow this procedure internally before writing:\n"
        "  STEP 1 — Entity sweep. Identify EVERY salient detail in the article: "
        "people (with full names, roles, titles), organizations and institutions, "
        "places, dates and times, numbers, money figures, percentages, "
        "laws and regulations, products, named events, direct quotes, "
        "stated causes, stated consequences, and any qualifications or caveats.\n"
        "  STEP 2 — Draft a briefing that incorporates every item from STEP 1.\n"
        "  STEP 3 — Verify: re-check the article and confirm no salient detail is missing. "
        "If anything is missing, fuse it in. Do NOT drop a detail to keep the text short.\n\n"

        "Output style for the final briefing:\n"
        "  - Length is dictated ENTIRELY by the article's information density. There is NO upper limit "
        "and NO target sentence count. A long, fact-rich article MUST produce a long briefing. "
        "If you find yourself compressing, summarizing, or skipping any named person, organization, place, "
        "date, number, quote, cause, or consequence — STOP and rewrite to include it. "
        "Err on the side of more detail, never less.\n"
        "  - Preserve specific numbers, dates, full names, titles, and direct quotes VERBATIM.\n"
        "  - Preserve the sequence of events and cause-effect chain.\n"
        "  - Include attribution when the article gives it (\"X said\", \"according to Y\").\n"
        "  - Flowing prose. No bullet points, no headings, no markdown.\n"
        "  - Do NOT add information that is not in the source. Do NOT speculate or infer. "
        "Length must come from the source's actual content, not from padding.\n\n"

        "Return one entry per article using the article's index.\n\n"
        f"Articles:\n\n{payload}"
    )

    response = gemini_generate(prompt, response_schema=schema)
    data = parse_json_lenient(response.text)

    out = []
    for entry in data.get("summaries", []):
        i = entry.get("index", -1)
        if 0 <= i < len(articles):
            out.append({
                "source": source_name,
                "title": articles[i]["title"],
                "url": articles[i]["url"],
                "summary": entry["summary"],
            })
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
        "Group the news items below so each group covers the same underlying news event. "
        "Items about the same story (even from different sources or in different languages) belong together. "
        "Items unique to one event form a single-item group. "
        "Every index must appear in exactly ONE group. No duplicates, no omissions.\n\n"
        f"{payload}"
    )

    response = gemini_generate(prompt, response_schema=schema)
    data = parse_json_lenient(response.text)
    return [c["indices"] for c in data.get("clusters", [])]


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
        "These items report the SAME news event from different sources. "
        "Produce TWO outputs in one response:\n"
        "  1. title — a short, neutral, headline-style title for the event (≤15 words, no clickbait).\n"
        "  2. summary — ONE unified, COMPREHENSIVE briefing preserving every salient detail across all sources. "
        "Coverage beats brevity.\n\n"

        "Procedure (internal) for the summary:\n"
        "  STEP 1 — Detail sweep. List every distinct salient detail mentioned by ANY item: "
        "people (full names, roles, titles), organizations, places, dates and times, "
        "numbers, money figures, percentages, laws, products, named events, direct quotes, "
        "stated causes, stated consequences, qualifications, and caveats.\n"
        "  STEP 2 — Draft a unified briefing that includes EVERY detail from STEP 1. "
        "If two sources disagree on a fact, present both versions with attribution "
        "(e.g., 'Reuters reports 50 dead while AP reports 47').\n"
        "  STEP 3 — Verify: re-check each item and confirm no salient detail is missing. "
        "If only one source mentions a detail, KEEP IT.\n\n"

        "Style for the summary:\n"
        "  - Length is dictated by the COMBINED information density of all sources. There is NO upper limit "
        "and NO target sentence count. If multiple sources together provide many distinct facts, "
        "the merged briefing MUST be long. Err on the side of more detail, never less. "
        "If you catch yourself compressing or dropping a detail any source mentioned, rewrite to include it.\n"
        "  - Preserve numbers, dates, full names, titles, and direct quotes VERBATIM.\n"
        "  - Preserve sequence of events and cause-effect chain.\n"
        "  - Flowing prose. No bullets, no headings, no markdown.\n"
        "  - Language for BOTH title and summary: use the language of the MAJORITY of items.\n"
        "  - Do NOT invent details. Do NOT speculate. Length must come from the sources' actual content.\n\n"

        f"Items:\n\n{payload}"
    )

    response = gemini_generate(prompt, response_schema=schema)
    data = parse_json_lenient(response.text)
    return data["title"], data["summary"]


def main():
    articles_by_source = collect_articles()

    all_summaries = []
    for source, articles in articles_by_source.items():
        print(f"Summarizing {len(articles)} articles from {source}...")
        all_summaries.extend(summarize_source(source, articles))

    if not all_summaries:
        print("No articles collected.")
        return

    print(f"Clustering {len(all_summaries)} summaries...")
    clusters = cluster_summaries(all_summaries)

    cluster_items = []
    for cluster_indices in clusters:
        items = [all_summaries[i] for i in cluster_indices if 0 <= i < len(all_summaries)]
        if items:
            cluster_items.append(items)

    from collections import Counter
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
        title, merged = merge_cluster(items)
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

    from datetime import date
    report_text = build_report_text(output)
    filename = f"Haber Analizi {date.today().strftime('%d %m %Y')}.md"
    messages = build_discord_messages(output)

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook:
        print(f"Posting {len(messages)} chunked messages to Discord webhook...")
        send_to_discord(messages, webhook)
        upload_report_to_discord(report_text, filename, webhook)
    else:
        print("DISCORD_WEBHOOK_URL not set in env — skipping Discord. Add it to keys/.env to enable.")


def build_report_text(output):
    """Return the full Markdown report as a single string (no file I/O)."""
    from datetime import date

    multi = [e for e in output if len(e["sources"]) > 1]
    singles = [e for e in output if len(e["sources"]) == 1]

    by_source = {}
    for entry in singles:
        src = entry["sources"][0]["source"]
        by_source.setdefault(src, []).append(entry)

    lines = [f"# News Report — {date.today().isoformat()}", ""]

    if multi:
        lines.append("# Common News")
        lines.append("*Stories covered by multiple sources*")
        lines.append("")
        for i, entry in enumerate(multi, 1):
            lines.append(f"## {i}. {entry['title']}")
            lines.append("")
            lines.append(entry["summary"])
            lines.append("")
            lines.append("**Sources:**")
            for s in entry["sources"]:
                lines.append(f"- [{s['source']} — {s['title']}]({s['url']})")
            lines.append("")
            lines.append("---")
            lines.append("")

    for src in RSS_FEEDS:
        if src not in by_source:
            continue
        lines.append(f"# {src}")
        lines.append("")
        for entry in by_source[src]:
            url = entry["sources"][0]["url"]
            lines.append(f"## {entry['title']}")
            lines.append("")
            lines.append(entry["summary"])
            lines.append("")
            lines.append(f"[Read more]({url})")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


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
        tail = " · ".join(s["source"] for s in sources)
        header = f"## [{title}](<{primary_url}>)\n-# {primary_source} · Covered by: {tail}\n\n"
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
    from datetime import date

    multi = [e for e in output if len(e["sources"]) > 1]
    singles = [e for e in output if len(e["sources"]) == 1]

    by_source = {}
    for entry in singles:
        src = entry["sources"][0]["source"]
        by_source.setdefault(src, []).append(entry)

    first = _section_header(f"📰 News Report — {date.today().isoformat()}",
                            subtitle=f"{len(multi)} cross-source stories · {len(singles)} unique stories")
    first["content"] = f"<@331085172264927235>\n{first['content']}"
    first["allowed_mentions"] = {"parse": ["users"]}
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


def send_to_discord(payloads, webhook_url, delay=0.4):
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


def upload_report_to_discord(report_text, filename, webhook_url):
    """POST a final message with the report attached as a .md file. No ping (already fired)."""
    print(f"Uploading {filename} ({len(report_text):,} chars) to Discord...")
    payload = {"content": "📄 **Full report (Markdown):**"}
    files = {
        "payload_json": (None, json.dumps(payload), "application/json"),
        "file": (filename, report_text.encode("utf-8"), "text/markdown"),
    }
    try:
        r = requests.post(webhook_url, files=files, timeout=60)
        if r.status_code >= 400:
            print(f"  Discord HTTP {r.status_code}: {r.text[:300]}")
        else:
            print(f"  Discord upload OK")
    except Exception as e:
        print(f"  Discord upload error: {e}")


if __name__ == "__main__":
    main()
