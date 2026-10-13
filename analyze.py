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


# Relevance test for the post-summary filter. The LLM reads this verbatim.
# One governing question (<decision>), six channels of relevance (<keep>), the drop
# categories (<drop>) and real reader decisions as calibration pairs (<calibration>).
# There is no "when unsure, keep": a story whose connection cannot be named is dropped.
INTERESTS = """
<reader>
The reader lives in Türkiye and follows the news to see what can reach them: Türkiye itself, its neighbourhood, the wars of the day, the big powers whose decisions ripple across the world, and the shifts in ideas and societies that show where the world is heading. A story earns its place only through one of those connections. A famous name, a dramatic event or a high casualty count does not count by itself.
</reader>

<decision>
For every item, answer this question in one sentence: "What does this story tell the reader about Türkiye, about a war, about a big power, or about where the world is heading?"
If you can answer with a concrete point, KEEP. If the honest answer is "nothing beyond the event itself", DROP. There is no "unsure" bucket: a story whose connection you cannot name in one sentence is dropped.
</decision>

<keep>
A story is relevant when at least one of these holds.

1. TÜRKİYE, at the national level. Government and opposition moves, parliament, court cases of national weight, investigations and scandals, the economy (lira, interest rates, inflation, markets, budget, trade, national data releases, major sectors), foreign policy, security, major regulation, large Turkish companies. Security and counter-terrorism operations against terrorist organisations (PKK, DHKP-C, ISIS and others), whatever their size. Mass-scale incidents inside Türkiye that affect thousands of people at once, such as a mass illness or a large disaster. Also any foreign story in which Türkiye, a Turkish company or Turkish interests are an actor.

2. WAR AND CONFLICT, anywhere, at any scale, as long as the story is about the conflict: fighting, strikes, casualties, ceasefires, negotiations, war aims, weapons, war finance and sanctions, spies and sabotage, the fate of the people inside it, and analyses of how a war is going or could spread. Türkiye is affected by every war, so this channel is wide.

3. BIG POWERS: the USA, China, Russia, the EU and its large members, the UK, India, Israel, Iran, Saudi Arabia. A nationwide event or a national-level decision in one of these is relevant because it changes how that power acts: elections and the public mood behind them, the leader's conduct and use of power, major court rulings against the government, nationwide disasters, the labour market and the economy, big regulation, government crises. Features and analyses count when they read the direction of the country (its mood, its voters, its system, its surveillance), not when they profile one person.

4. SHIFTS IN IDEAS AND PRECEDENTS: an event that changes what is treated as legitimate or normal, even when the actors are small. A state denying another country's legitimacy, a party declaring an ideology racist, a precedent-setting ruling on rights, speech, religion or borders, a first-of-its-kind law with wider meaning. The size of the actor does not matter; the size of the shift does.

5. CRACKS IN MODEL SOCIETIES: a developed, stable country showing a structural problem. Violence or abuse inside an institution or a sector, a welfare state failing a group, a trusted system found broken. These show where developed societies are heading.

6. THE GLOBAL ECONOMY: oil and gas, gold, the dollar, bond markets, central banks, trade rules and quotas, sanctions, the IMF and the World Bank, AI's effect on jobs, climate milestones. Anything that moves prices or policy in Türkiye.

7. EVENTS OF THE WHOLE WORLD: the few events the entire world follows at the same moment wherever they happen, such as the Nobel prizes, decisions of the UN Security Council or General Assembly, and the G7, G20 and COP summits.
</keep>

<drop>
A story is irrelevant when it is about the event alone and nothing beyond it. Drop these even when the source is prestigious or the casualty count is high.

- PERSONAL MATTERS. A famous person's health, illness, family or private life. One official's or one businessperson's misconduct, wealth, taxes or scandal when no government decision and no public money is involved. One person's release, arrest, prison story, memoir or interview about their own experience.
- SINGLE INCIDENTS. One accident, one shipwreck, one crime, one attack, one arrest, one plot, one police incident, one school stabbing, when the story does not reach national politics. A single event inside a war zone still counts under WAR; a single event elsewhere does not. A counter-terrorism operation in Türkiye is not a single incident; an ordinary crime operation (loan sharks, drugs, prostitution) is.
- LOCAL AND SMALL-COUNTRY DOMESTIC MATTERS with no shift in ideas: a draft law, a fuel-cost complaint, a sector dispute, a party's internal reshuffle, a provincial problem. This includes local incidents in Türkiye (a poisoning of a few people, a fire, a court step) that carry no national political or economic weight. When thousands of people are affected, the incident is national and is kept.
- NATURAL DISASTERS outside the big powers. A hurricane in a mid-sized country stays that country's problem; a hurricane across the US is a big-power event.
- ONE CANDIDATE, ONE VOICE. A single candidate's interview or campaign line, a single expert's views on a sector, a tech executive's opinions. The state of an election is relevant; one voice in it is not.
- NO NEW EVENT. Explainers, history pieces, opinion columns about a person's character, timelines that retell known facts, daily agendas and schedules of upcoming events, market notes that only restate a price without a cause.
- NOISE. Sports, celebrity and entertainment, lifestyle, recipes, consumer tips, product reviews, astrology, clickbait, routine personnel appointments, routine corporate filings, routine weather.
</drop>

<calibration>
These pairs show where the line is. Each one is a real decision by the reader.
KEEP: India's opposition and youth protest a voter-roll purge of 130 million names. A nationwide event that shows where India is going.
DROP: A man is detained in Poland for plotting to kill the prime minister. One arrest, nothing beyond it.
KEEP: Hurricane Isaias cuts power to 800,000 homes in Florida and Alabama. A nationwide-scale event in a big power.
DROP: Hurricane Simon hits Mexico's Pacific coast. A disaster that stays Mexico's.
KEEP: 72 gold miners are killed in an air strike on the Sudan-Egypt border. Part of a war.
DROP: 139 migrants drown off Djibouti. A single shipwreck.
KEEP: A judge halts the Trump administration's $20m taxpayer-funded TV ads. The president's conduct and public money.
DROP: Trump weighs firing an ambassador accused of taking gifts. One official's misconduct.
KEEP: China sanctions Paraguayan and Belizean ministers over Taiwan ties. A power denying a country's legitimacy; the actors are small, the shift is not.
DROP: Denmark plans to let teachers restrain violent pupils. A small country's school rule with no wider shift.
KEEP: The UK Greens declare Zionism racism and then lose a by-election. An ideological move and its setback.
DROP: A column argues that Netanyahu uses antisemitism charges to silence critics. Opinion about one person, no event.
KEEP: Women in French e-sports describe sexual violence across the sector. A developed society showing a structural problem.
DROP: A freed journalist describes his 16 months in a Russian prison. One person's experience.
KEEP: A reporter returns to China and finds a richer, more paranoid surveillance state. Reads the direction of a big power.
DROP: The Pope will have lung surgery. A famous person's health, with no political content.
KEEP: Nebraska ranchers squeezed by tariffs and diesel question their loyalty to Trump. The mood of a big power's voters.
DROP: A Michigan Senate candidate says Democrats must win big. One candidate's line.
KEEP: A profile of a captured Russian spy who regrets only being caught. Part of the war.
KEEP: The Nobel economics prize goes to Robert Hall and Michael Woodford. The whole world follows the Nobel prizes.
KEEP: A counter-terrorism operation against DHKP-C detains 18 people in six provinces. An operation against a terrorist organisation in Türkiye.
DROP: A loan-sharking operation in İzmir catches 13 suspects. Ordinary crime, one operation.
KEEP: About 5,000 people in Tokat seek hospital treatment for a stomach illness. A mass-scale incident in Türkiye.
DROP: 16 people in Antalya are poisoned by botox injections. A local incident in Türkiye with no national weight.
DROP: "Economy and politics agenda for the day". A schedule, not news.
</calibration>
""".strip()


# Up to 5 API keys read from env: GEMINI_API_KEY_1 ... GEMINI_API_KEY_5
# Falls back to GEMINI_API_KEY if numbered keys are not set.
# Models are tried in this order. Every call starts on the first model, rotating through the
# keys; when EVERY key's free-tier daily quota on that model is spent, the run moves to the next
# model. Google counts the daily quota per model, so the same keys get a fresh allowance there.
# Override with GEMINI_MODELS="modelA,modelB,..." when Google ships a newer Flash.
MODEL_CHAIN = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash").split(",") if m.strip()]
PRIMARY_MODEL = MODEL_CHAIN[0]

_raw_keys = {f"GEMINI_API_KEY_{i}": os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 6)}
GEMINI_KEYS = [v for v in _raw_keys.values() if v]
if not GEMINI_KEYS:
    fallback = os.environ.get("GEMINI_API_KEY")
    if fallback:
        GEMINI_KEYS = [fallback]
print(f"Loaded {len(GEMINI_KEYS)} Gemini API key(s): " +
      (", ".join(name for name, val in _raw_keys.items() if val) or "(fallback GEMINI_API_KEY)"))
print("Models (each used until every key's daily quota on it is spent): " + " -> ".join(MODEL_CHAIN))


class GeminiError(RuntimeError):
    """Base for errors gemini_generate raises after giving up."""


class ContentBlocked(GeminiError):
    """The model refused the prompt (PROHIBITED_CONTENT / safety). Deterministic —
    the same prompt fails on every key, so callers should shrink the prompt, not retry."""


class QuotaExhausted(GeminiError):
    """Every usable key is out of daily quota (or dead). Nothing more will succeed today."""


_key_lock = threading.Lock()
_key_counter = 0
# key_idx (1-based) -> reason. A key lands here when it can never succeed again this run on ANY
# model: 403 API_KEY_SERVICE_BLOCKED or an invalid key.
_dead_keys = {}
# (model, key_idx) -> True once that key's free-tier DAILY quota for that model is used up.
# Quota is per model, so the key stays usable on the other model in MODEL_CHAIN.
_quota_dead = {}
_key_ok_calls = Counter()   # (model, key_idx) -> successful calls, for the end-of-run summary


def _rotated_keys(model):
    """[(key_idx_1based, key), ...] of keys still usable on `model`, rotated so a different
    key starts each call."""
    global _key_counter
    if not GEMINI_KEYS:
        return []
    with _key_lock:
        start = _key_counter % len(GEMINI_KEYS)
        _key_counter += 1
    n = len(GEMINI_KEYS)
    order = [(((start + i) % n) + 1, GEMINI_KEYS[(start + i) % n]) for i in range(n)]
    return [(idx, key) for idx, key in order
            if idx not in _dead_keys and (model, idx) not in _quota_dead]


def _kill_key(key_idx, reason, model=None):
    """Retire a key: for every model (model=None: blocked/invalid key) or only for `model`
    (its daily quota on that model is spent)."""
    n = len(GEMINI_KEYS)
    with _key_lock:
        if model is None:
            if key_idx in _dead_keys:
                return
            _dead_keys[key_idx] = reason
        else:
            if (model, key_idx) in _quota_dead:
                return
            _quota_dead[(model, key_idx)] = True
    if model is None:
        live = n - len(_dead_keys)
        print(f"  !! key {key_idx}/{n} retired for the rest of this run ({reason}). {live} key(s) left.")
    else:
        live = len(_rotated_keys_quiet(model))
        print(f"  !! key {key_idx}/{n} out of daily quota on '{model}'. {live} key(s) left on this model.")


def _rotated_keys_quiet(model):
    """Same filter as _rotated_keys without advancing the rotation counter."""
    return [(i + 1, k) for i, k in enumerate(GEMINI_KEYS)
            if (i + 1) not in _dead_keys and (model, i + 1) not in _quota_dead]


def _dead_summary(model):
    bits = []
    for idx in range(1, len(GEMINI_KEYS) + 1):
        if idx in _dead_keys:
            bits.append(f"key {idx}: {_dead_keys[idx]}")
        elif (model, idx) in _quota_dead:
            bits.append(f"key {idx}: daily quota used up")
    return "; ".join(bits)


def gemini_key_report():
    """One-line status of every key, printed at the end of the run."""
    parts = []
    for idx in range(1, len(GEMINI_KEYS) + 1):
        if idx in _dead_keys:
            state = _dead_keys[idx]
        else:
            spent = [m for m in MODEL_CHAIN if (m, idx) in _quota_dead]
            state = ("quota gone on " + ", ".join(spent)) if spent else "ok"
        calls = ", ".join(f"{_key_ok_calls[(m, idx)]} on {m}" for m in MODEL_CHAIN)
        parts.append(f"key {idx}: {calls}, {state}")
    return "; ".join(parts) if parts else "no keys"


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


_TRANSIENT_SIGNALS = (
    "503", "unavailable", "overloaded", "high demand", "502", "504", "500", "internal",
    "499", "deadline", "timeout", "timed out", "cancelled", "canceled",
    "server disconnected", "remoteprotocolerror", "connection reset", "connection aborted",
    "connection closed", "broken pipe", "incomplete read", "connecterror", "readerror",
)


def _classify_error(e):
    """Map an exception from generate_content to (kind, detail).

    kind is one of:
      'blocked'   - content refusal; retrying the same prompt is pointless
      'notfound'  - model id unknown; abort the run
      'dead'      - this key can never work (403 service blocked, invalid key)
      'daily'     - this key's free-tier DAILY request quota is spent
      'rate'      - per-minute rate limit; wait `detail` seconds and try again
      'transient' - Google-side 5xx / network; rotate and retry with backoff
      'fatal'     - anything else; raise immediately
    Uses the structured fields google-genai puts on APIError (code/status/details) and
    falls back to substring matching for plain exceptions."""
    if isinstance(e, ContentBlocked):
        return "blocked", str(e)

    code = getattr(e, "code", None)
    status = str(getattr(e, "status", "") or "").upper()
    message = str(getattr(e, "message", "") or "")
    details = getattr(e, "details", None)
    details_text = json.dumps(details, default=str) if details is not None else ""
    blob = f"{status} {message} {details_text} {e}".lower()

    if code == 404 or status == "NOT_FOUND":
        return "notfound", _short_err(e)
    if code == 403 or status == "PERMISSION_DENIED":
        return "dead", f"403 {message or 'permission denied'}"
    if code == 400 and "api key" in blob:
        return "dead", f"400 {message}"
    if code == 429 or status == "RESOURCE_EXHAUSTED" or "quota" in blob or "rate limit" in blob:
        if re.search(r"perday|per day|daily", blob):
            return "daily", "free-tier daily request quota used up"
        m = re.search(r"retrydelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", blob)
        wait = float(m.group(1)) if m else 20.0
        return "rate", min(max(wait, 5.0), 90.0)
    if code is not None and 500 <= int(code) < 600:
        return "transient", _short_err(e)
    if any(s in blob for s in _TRANSIENT_SIGNALS):
        return "transient", _short_err(e)
    return "fatal", _short_err(e)


def _build_config(model, response_schema):
    thinking = types.ThinkingConfig(thinking_level="high")
    if response_schema is not None:
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=thinking,
        )
    return types.GenerateContentConfig(thinking_config=thinking)


def _check_response(response):
    """Raise ContentBlocked for a safety/prohibited-content refusal, RuntimeError for any other empty reply."""
    if response.text:
        return
    finish, block = "?", "?"
    try:
        if response.candidates:
            finish = str(response.candidates[0].finish_reason or "?")
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block = str(response.prompt_feedback.block_reason)
    except Exception:
        pass
    tag = f"{finish} {block}".upper()
    if any(s in tag for s in ("PROHIBITED", "SAFETY", "BLOCKLIST", "SPII", "RECITATION")):
        raise ContentBlocked(f"model refused the prompt (finish={finish}, block={block})")
    raise RuntimeError(f"empty response from model (finish={finish}, block={block})")


MAX_ROUNDS = 3               # passes over the live keys before giving up on transient errors
TRANSIENT_RETRY_WAIT = 20    # base seconds between rounds; multiplied by round number


def gemini_generate(contents, response_schema=None):
    """Run the prompt on the first model in MODEL_CHAIN; once every key's daily quota on it is
    spent, on the next one, and so on. Raises QuotaExhausted only when no model has a usable key."""
    if not GEMINI_KEYS:
        raise RuntimeError("No Gemini API keys configured. Set GEMINI_API_KEY_1..5 or GEMINI_API_KEY.")

    last_quota_err = None
    for pos, model in enumerate(MODEL_CHAIN):
        if not _rotated_keys_quiet(model):
            # Already known to be spent — skip without spending a request.
            last_quota_err = last_quota_err or QuotaExhausted(
                f"No usable Gemini key left on '{model}': {_dead_summary(model)}")
            continue
        try:
            return _generate_on_model(model, contents, response_schema)
        except QuotaExhausted as e:
            last_quota_err = e
            if pos + 1 < len(MODEL_CHAIN):
                print(f"  >> '{model}' quota is gone on every key; switching to "
                      f"'{MODEL_CHAIN[pos + 1]}' for the rest of this run.")
            continue

    raise QuotaExhausted(
        "All Gemini keys are dead or out of daily quota on every model ("
        + ", ".join(MODEL_CHAIN) + f"). {last_quota_err}"
    )


def _generate_on_model(model, contents, response_schema):
    """Run `model` across the keys still usable on it.

    Per key: content refusals raise ContentBlocked at once (no rotation — every key would refuse).
    A 403 / invalid key retires that key for every model; a spent daily quota retires it for this
    model only. Per-minute 429s and 5xx errors rotate to the next key and, if the whole round fails
    that way, wait and try again (up to MAX_ROUNDS rounds).
    Raises QuotaExhausted when no key is left on this model."""
    config = _build_config(model, response_schema)
    n = len(GEMINI_KEYS)
    last_err = None

    for round_no in range(1, MAX_ROUNDS + 1):
        keys = _rotated_keys(model)
        if not keys:
            raise QuotaExhausted(f"No usable Gemini key left on '{model}': {_dead_summary(model)}")

        round_retryable = 0
        rate_wait = 0.0
        for key_idx, key in keys:
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
                _check_response(response)
                _key_ok_calls[(model, key_idx)] += 1
                return response
            except Exception as e:
                last_err = e
                kind, detail = _classify_error(e)
                tag = f"[{model} | key {key_idx}/{n}]"
                if kind == "blocked":
                    raise
                if kind == "notfound":
                    raise RuntimeError(f"Model '{model}' not available on these keys: {detail}")
                if kind == "dead":
                    _kill_key(key_idx, detail)
                    continue
                if kind == "daily":
                    _kill_key(key_idx, detail, model=model)
                    continue
                if kind == "rate":
                    print(f"  {tag} per-minute rate limit, rotating (retry after ~{detail:.0f}s)")
                    rate_wait = max(rate_wait, detail)
                    round_retryable += 1
                    continue
                if kind == "transient":
                    print(f"  {tag} transient error, rotating: {detail}")
                    round_retryable += 1
                    continue
                print(f"  {tag} non-retryable error: {detail}")
                raise

        if not _rotated_keys_quiet(model):
            break
        if round_retryable and round_no < MAX_ROUNDS:
            wait = max(TRANSIENT_RETRY_WAIT * round_no, rate_wait)
            print(f"  '{model}' round {round_no}/{MAX_ROUNDS}: all live keys failed with retryable errors. "
                  f"Waiting {wait:.0f}s...")
            time.sleep(wait)

    if not _rotated_keys_quiet(model):
        raise QuotaExhausted(
            f"All Gemini keys are dead or out of daily quota on '{model}': {_dead_summary(model)}. "
            f"Last error: {_short_err(last_err) if last_err else '?'}"
        )
    raise GeminiError(
        f"'{model}' failed on every live key for {MAX_ROUNDS} rounds (Google-side errors / rate limits). "
        f"Last error: {_short_err(last_err) if last_err else '?'}"
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
            out.extend(_summarize_batch_bisecting(source_name, batch))
        except QuotaExhausted as e:
            print(f"  ! [{source_name}] batch {batch_idx}/{len(batches)} skipped: {e}")
            break
        except Exception as e:
            print(f"  ! [{source_name}] batch {batch_idx}/{len(batches)} failed, "
                  f"skipping its {len(batch)} article(s): {e}")
    return out


def _summarize_batch_bisecting(source_name, articles):
    """Summarize a batch; if the model refuses the whole batch (PROHIBITED_CONTENT), split it in half
    and retry each half so one offending article costs one article, not ten."""
    try:
        return _summarize_batch(source_name, articles)
    except ContentBlocked as e:
        if len(articles) == 1:
            print(f"  ! [{source_name}] model refused this article, skipping: {articles[0]['title'][:70]} ({e})")
            return []
        mid = len(articles) // 2
        print(f"  ! [{source_name}] model refused a batch of {len(articles)}; splitting to find the offending article")
        return (_summarize_batch_bisecting(source_name, articles[:mid])
                + _summarize_batch_bisecting(source_name, articles[mid:]))


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
        "  STEP 1 — Identify the core of the story: what happened, who did it, and why it matters.\n"
        "  STEP 2 — Pick ONLY the most important supporting details: the one or two decisive numbers, "
        "dates, names or quotes without which the story cannot be understood. Everything else "
        "(secondary actors, background history, reaction quotes, minor figures, procedural steps) "
        "is left out.\n"
        "  STEP 3 — Write a short briefing: the core first, then the chosen details.\n\n"

        "Constraints:\n"
        "  - Short and essential. A reader wants the gist, not a full account.\n"
        "  - The central numbers, dates and names you do keep must be VERBATIM.\n"
        "  - Keep the main cause-effect link if there is one.\n"
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
    except QuotaExhausted as e:
        print(f"  ! Clustering skipped — no Gemini quota left ({e}).")
        print("  ! Every story will be reported on its own. Cross-source stories cannot be connected today.")
        return [[i] for i in range(len(summaries))]
    except Exception as e:
        print(f"  ! Clustering failed ({e}); treating every summary as its own story.")
        return [[i] for i in range(len(summaries))]


MERGE_BATCH_SIZE = 6  # clusters merged per Gemini call; free tier allows ~20 requests/day/key


def _merge_prompt_rules():
    return (
        "For EACH group produce TWO outputs:\n"
        "  1. title — a headline (≤15 words) that pulls the reader in while staying honest. "
        "Be specific: concrete actor + concrete action + the story's most striking element "
        "(unexpected outcome, contradiction, number, confrontation), taken from the sources. "
        "Active verb, key fact front-loaded. Curiosity is fine but the title's promise must be "
        "answered in the summary: no withholding phrases ('what happened next', 'shocking decision'), "
        "no ALL-CAPS or exclamations, no unanswered question headlines, no hype adjectives — "
        "the striking element must come from the story's own facts.\n"
        "  2. summary — ONE unified briefing that merges the group's items into a single readable account.\n\n"

        "How to merge:\n"
        "  - The items in a group overlap heavily. State the core of the story ONCE — never stack "
        "near-identical tellings of the same fact from different sources.\n"
        "  - Keep ONLY the most important details: the decisive numbers, dates, names or quotes "
        "that the story cannot be understood without. Leave out secondary actors, background history, "
        "reaction quotes, minor figures and procedural steps, even if a source mentions them.\n"

        "  - If sources disagree on a fact, give both versions with attribution "
        "(e.g., 'Reuters reports 50 dead while AP reports 47').\n"
        "  - Open with whatever matters most in THIS story — judge per story; sometimes it is a person, "
        "sometimes a number, sometimes a decision — then continue in decreasing importance.\n\n"

        "Constraints:\n"
        "  - Never mix facts between groups. Each group is a separate story.\n"
        "  - The numbers, dates, names and quotes you keep must be VERBATIM.\n"
        "  - Keep the main cause-effect link.\n"
        "  - Flowing prose. No bullets, no headings, no markdown.\n"
        "  - Language for BOTH title and summary of a group: use the language of the MAJORITY of that "
        "group's items; if tied, use Turkish.\n"
        "  - Do NOT invent details. Do NOT speculate. Short beats complete.\n"
        "  - Return exactly one entry per group, using the group's number."
    )


def _stacked(items):
    """Fallback when the LLM cannot merge: keep the first title and stack the source summaries."""
    return items[0]["title"], "\n\n".join(f"({it['source']}) {it['summary']}" for it in items)


def merge_clusters(cluster_list):
    """Stage 3: merge multi-source clusters into (title, summary) pairs, several clusters per Gemini
    call to stay inside the free-tier daily request quota. Returns a list aligned with cluster_list.
    Singletons never touch the LLM; a failed call falls back to stacking that batch's summaries."""
    results = [None] * len(cluster_list)
    multi_idx = [i for i, items in enumerate(cluster_list) if len(items) > 1]
    for i, items in enumerate(cluster_list):
        if len(items) == 1:
            results[i] = (items[0]["title"], items[0]["summary"])

    schema = {
        "type": "object",
        "properties": {
            "merged": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group": {"type": "integer"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["group", "title", "summary"],
                },
            },
        },
        "required": ["merged"],
    }

    batches = [multi_idx[i:i + MERGE_BATCH_SIZE] for i in range(0, len(multi_idx), MERGE_BATCH_SIZE)]
    quota_gone = False
    for b_no, batch in enumerate(batches, 1):
        for i in batch:
            print(f"  [merge {b_no}/{len(batches)}] group {i}: {len(cluster_list[i])} sources: "
                  + ", ".join(it["source"] for it in cluster_list[i]))
        if quota_gone:
            for i in batch:
                results[i] = _stacked(cluster_list[i])
            continue

        groups_text = []
        for g_no, i in enumerate(batch, 1):
            body = "\n\n".join(f"({it['source']}) {it['title']}\n{it['summary']}" for it in cluster_list[i])
            groups_text.append(f"=== GROUP {g_no} ===\n{body}")
        prompt = (
            f"{len(batch)} groups of news items follow. Each group's items report the SAME news event "
            f"from different sources.\n\n" + "\n\n".join(groups_text) + "\n\n" + _merge_prompt_rules()
        )

        try:
            response = gemini_generate(prompt, response_schema=schema)
            data = parse_json_lenient(response.text)
            got = {}
            for entry in data.get("merged", []):
                g = entry.get("group")
                if isinstance(g, int) and 1 <= g <= len(batch) and entry.get("summary"):
                    got[g] = (entry.get("title") or cluster_list[batch[g - 1]][0]["title"], entry["summary"])
            for g_no, i in enumerate(batch, 1):
                if g_no in got:
                    results[i] = got[g_no]
                else:
                    print(f"  ! merge batch {b_no}: no result for group {g_no}; stacking its summaries unmerged.")
                    results[i] = _stacked(cluster_list[i])
        except QuotaExhausted as e:
            print(f"  ! Merge batch {b_no}/{len(batches)} skipped, quota gone ({e}); stacking the rest unmerged.")
            quota_gone = True
            for i in batch:
                results[i] = _stacked(cluster_list[i])
        except Exception as e:
            print(f"  ! Merge batch {b_no}/{len(batches)} failed ({e}); stacking its summaries unmerged.")
            for i in batch:
                results[i] = _stacked(cluster_list[i])
    return results


def filter_summaries(summaries):
    """Drop summaries that fail the INTERESTS test. Falls back to keeping all on filter failure
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
            "drop": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["index", "reason"],
                },
            },
        },
        "required": ["drop"],
    }

    prompt = (
        f"<items>\n{payload}\n</items>\n\n"
        f"{INTERESTS}\n\n"
        "Apply <decision> to every item in <items>. Return drop: one entry per item you DROP, with its "
        "index and a one-sentence reason naming the <drop> category it falls under. Items you do not "
        "list are kept. Return only drop."
    )

    print(f"Filtering {len(summaries)} summaries against the relevance test...")
    try:
        response = gemini_generate(prompt, response_schema=schema)
        data = parse_json_lenient(response.text)
        drop = {}
        for entry in data.get("drop", []):
            if isinstance(entry, dict):
                i, why = entry.get("index", -1), (entry.get("reason") or "").strip()
            else:
                i, why = entry, ""
            if isinstance(i, int) and 0 <= i < len(summaries):
                drop[i] = why
    except Exception as e:
        print(f"  Filter call failed ({e}); keeping all.")
        return summaries

    kept = [s for i, s in enumerate(summaries) if i not in drop]
    print(f"  Kept {len(kept)}/{len(summaries)} ({len(drop)} dropped)")
    if drop:
        print("  Dropped:")
        for i in sorted(drop):
            why = f" — {drop[i]}" if drop[i] else ""
            print(f"    - [{summaries[i]['source']}] {summaries[i]['title']}{why}")
    return kept


# PIPELINE_MODE selects what this process does:
#   full   (default) scrape -> Gemini summaries/filter/cluster/merge -> Discord
#   scrape           scrape only; write the raw articles to ARTICLES_FILE and stop (no Gemini)
#   post             read a ready-made CLUSTERS_FILE (same shape as news_clusters.json) and post it
# "scrape" + "post" let the summarizing be done outside this process when Gemini is unusable.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "full").strip().lower() or "full"
ARTICLES_FILE = os.environ.get("ARTICLES_FILE", "articles.json")
CLUSTERS_FILE = os.environ.get("CLUSTERS_FILE", "news_clusters.json")


def scrape_only():
    """Fetch every feed and dump the article texts; Gemini is never touched."""
    articles_by_source = collect_articles()
    total = sum(len(v) for v in articles_by_source.values())
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles_by_source, f, ensure_ascii=False, indent=1)
    print(f"Wrote {total} articles from {len(articles_by_source)} sources to {ARTICLES_FILE}")


def validate_clusters(output):
    """Shape check for an externally produced news_clusters.json."""
    if not isinstance(output, list) or not output:
        raise ValueError("clusters file must be a non-empty JSON list")
    for n, entry in enumerate(output):
        for key in ("title", "summary", "sources"):
            if not entry.get(key):
                raise ValueError(f"cluster {n} is missing '{key}'")
        for s in entry["sources"]:
            for key in ("source", "title", "url"):
                if not s.get(key):
                    raise ValueError(f"cluster {n} has a source missing '{key}'")
            if s["source"] not in RSS_FEEDS:
                raise ValueError(f"cluster {n}: unknown source name '{s['source']}'")


def post_clusters():
    """Post a ready-made clusters file through the normal Discord path."""
    with open(CLUSTERS_FILE, encoding="utf-8") as f:
        output = json.load(f)
    validate_clusters(output)
    multi = sum(1 for e in output if len(e["sources"]) > 1)
    print(f"Loaded {len(output)} clusters from {CLUSTERS_FILE} ({multi} multi-source)")
    with open("news_clusters.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    publish(output)


def publish(output):
    messages = build_discord_messages(output)

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if webhook:
        print(f"Posting {len(messages)} chunked messages to Discord webhook...")
        send_to_discord(messages, webhook)
        print("Pausing 10s before file upload to let Discord rate-limit window drain...")
        time.sleep(10)
        fname, text = build_report_file(output)
        upload_report_to_discord(
            fname,
            strip_markdown(text),
            webhook,
            f"**Haber Kaynakları {date.today().strftime('%d %m %Y')}**",
        )
    else:
        print("DISCORD_WEBHOOK_URL not set in env — skipping Discord. Add it to keys/.env to enable.")


def main():
    if PIPELINE_MODE == "scrape":
        return scrape_only()
    if PIPELINE_MODE == "post":
        return post_clusters()
    if PIPELINE_MODE != "full":
        raise SystemExit(f"Unknown PIPELINE_MODE '{PIPELINE_MODE}' (use full, scrape or post)")

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
    print(f"Gemini keys: {gemini_key_report()}")

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
    calls = (multi + MERGE_BATCH_SIZE - 1) // MERGE_BATCH_SIZE
    print(f"Merging {len(cluster_items)} clusters ({multi} multi-source in {calls} API call(s), {singletons} singletons skip)...")

    merged_pairs = merge_clusters(cluster_items)
    output = []
    for items, (title, merged) in zip(cluster_items, merged_pairs):
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

    publish(output)
    print(f"Gemini keys at end of run: {gemini_key_report()}")


def _is_turkish_cluster(entry):
    """A cluster counts as Turkish if more than half its sources are Turkish-language feeds.
    Used to sort multi-source stories: English clusters first, Turkish clusters last."""
    n_tr = sum(1 for s in entry["sources"] if s["source"] in TURKISH_SOURCES)
    return n_tr * 2 > len(entry["sources"])


REPORT_FILENAME = "News Report.md"


def build_report_file(output):
    """The whole report as one (filename, markdown) pair: cross-source stories first,
    then the single-source stories grouped by their feed."""
    multi = [e for e in output if len(e["sources"]) > 1]
    # Most-covered stories first; ties keep English clusters before Turkish ones.
    multi.sort(key=lambda e: (-len(e["sources"]), _is_turkish_cluster(e)))
    singles = [e for e in output if len(e["sources"]) == 1]

    by_source = {}
    for entry in singles:
        by_source.setdefault(entry["sources"][0]["source"], []).append(entry)

    lines = [
        f"# News Report — {date.today().isoformat()}",
        "",
        f"{len(multi)} cross-source stories · {len(singles)} unique stories",
        "",
        "---",
        "",
    ]

    if multi:
        lines += ["## Common News", ""]
        for i, entry in enumerate(multi, 1):
            lines += [f"### {i}. {entry['title']}", "", entry["summary"], "", "**Sources:**"]
            lines += [f"- [{s['source']} — {s['title']}]({s['url']})" for s in entry["sources"]]
            lines += ["", "---", ""]

    for src in RSS_FEEDS:
        if src not in by_source:
            continue
        lines += [f"## {src}", ""]
        for entry in by_source[src]:
            lines += [f"### {entry['title']}", "", entry["summary"], "",
                      f"[Read more]({entry['sources'][0]['url']})", "", "---", ""]

    return REPORT_FILENAME, "\n".join(lines)


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
        title = m.group(2).strip()
        underline = {1: "═", 2: "━"}.get(len(m.group(1)), "─") * min(len(title), 60)
        return f"\n{title}\n{underline}\n"

    text = re.sub(r'^(#{1,6})\s+(.+)$', heading_repl, text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\(<?([^)>]+)>?\)', r'\1', text)
    text = re.sub(r'^Sources:', 'SOURCES:', text, flags=re.MULTILINE)
    text = re.sub(r'^- ', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^(?!• )(.+)\n(?=• )', r'\1\n\n', text, flags=re.MULTILINE)
    text = re.sub(r'(^• .+)\n(?=• )', r'\1\n\n', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def upload_report_to_discord(filename, text, webhook_url, message):
    """Upload the whole report as one .md attachment. No ping (already fired)."""
    print(f"Uploading {filename} ({len(text):,} chars) to Discord...")
    payload = {"content": message}

    for attempt in range(1, 6):
        files = {
            "payload_json": (None, json.dumps(payload), "application/json"),
            "files[0]": (filename, text.encode("utf-8"), "text/markdown"),
        }
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
