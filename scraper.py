from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import os
import time
import urllib.request
import html as html_lib
import json
import re
import unicodedata
import requests
from curl_cffi import requests as cffi_requests
import trafilatura
from bs4 import BeautifulSoup

try:
    from newspaper import Article
except ImportError:
    Article = None

try:
    from googlenewsdecoder import gnewsdecoder
except ImportError:
    gnewsdecoder = None


# Add new sources to the matching dict below — RSS_FEEDS, TURKISH_SOURCES, and
# NON_ENGLISH_INTERNATIONAL_SOURCES are derived from these.
_INTERNATIONAL_FEEDS_EN = {
    "nytimes.com": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    # CNN's classic RSS feeds (rss.cnn.com/...) stopped updating years ago, so the 24h filter
    # never matched anything. The Google News sitemap is the index CNN still maintains
    # (listed in cnn.com/robots.txt). It covers every section, so it is path-filtered below.
    "cnn.com": "https://www.cnn.com/sitemap/news.xml",
    "bbc.com": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera.com": "https://www.aljazeera.com/xml/rss/all.xml",
    "npr.org": "https://feeds.npr.org/1001/rss.xml",
    "theguardian.com": "https://www.theguardian.com/world/rss",
    "propublica.org": "https://feeds.propublica.org/propublica/main",
    "theintercept.com": "https://theintercept.com/feed/?rss",
    "lemonde.fr/en": "https://www.lemonde.fr/en/rss/une.xml",
    "spiegel.de/international": "https://www.spiegel.de/international/index.rss",
}

# International sources whose titles are NOT in English. Titles from these
# get translated to English at summarization time so the report stays readable.
_INTERNATIONAL_FEEDS_NON_EN = {
    "welt.de": "https://www.welt.de/feeds/section/politik/ausland.rss",
}

_TURKISH_FEEDS = {
    "sozcu.com.tr": "https://www.sozcu.com.tr/feeds-rss-category-sozcu",
    "bloomberght.com": "https://www.bloomberght.com/rss",
    "veryansintv.com/direct": "https://www.veryansintv.com/feed",
    "bbc.com/turkce": "https://feeds.bbci.co.uk/turkce/rss.xml",
}

_INTERNATIONAL_FEEDS = {**_INTERNATIONAL_FEEDS_EN, **_INTERNATIONAL_FEEDS_NON_EN}

# International first, Turkish last — iteration order matters for the single-source report sections.
RSS_FEEDS = {**_INTERNATIONAL_FEEDS, **_TURKISH_FEEDS}
TURKISH_SOURCES = set(_TURKISH_FEEDS)
NON_ENGLISH_INTERNATIONAL_SOURCES = set(_INTERNATIONAL_FEEDS_NON_EN)

# Sitemap-style sources list the whole site. Keep only article URLs whose path matches, so
# sport/entertainment/video pages never cost an article fetch or a Gemini call.
SOURCE_URL_FILTERS = {
    "cnn.com": re.compile(
        r"cnn\.com/(?:\d{4}/\d{2}/\d{2}/)?"
        r"(?:world|politics|business|economy|middleeast|europe|asia|americas|africa|uk|china|india|australia|"
        r"science|climate|tech)/(?!live-news|videos?/)"
    ),
}

# Newest-first cap for sources that can list far more than a day's worth of relevant stories.
MAX_ARTICLES_PER_SOURCE = {
    "cnn.com": 25,
}

# Last resort for publishers that block direct fetches with 401/403 (nytimes.com does):
# a reader proxy that returns the page as plain text. Set READER_PROXY="" to disable.
READER_PROXY = os.environ.get("READER_PROXY", "https://r.jina.ai/")
READER_MIN_INTERVAL = 3.5   # seconds between reader calls (free tier allows ~20/min)
_last_reader_call = 0.0

# When even that fails, the feed's own description/summary is used as the article text
# if it is at least this long. The note below is prepended so the summarizer knows it is thin.
FEED_TEXT_MIN_CHARS = 80
FEED_TEXT_NOTE = ("[NOTE: The full article could not be fetched (publisher blocked the request). "
                  "Only the feed's own summary below is available; do not add anything beyond it.]")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}


BAD_TEXT_MARKERS = [
    "Sitemizdeki veriler Foreks",
    "YASAL UYARI",
    "Kayıtlı Elektronik Posta Adresi",
    "NASDAQ, Dow Jones Industrial Average",
    "Çerez Politikası",
    "Aydınlatma Metni",
    "Sitemizi kullanarak çerezleri kullanmamızı kabul edersiniz",
]

CHROME_SESSION = cffi_requests.Session(
    impersonate="chrome124",
)

CHROME_SESSION.cookies.update({
    "cookieconsent_status": "dismiss",
    "cookies_accepted": "1",
    "euConsent": "true",
    "gdpr": "0",
})

SAFARI_SESSION = cffi_requests.Session(
    impersonate="safari17_2_ios",
)

# Strategies tried in order. Each tuple: (session, extra kwargs).
# Chrome first (fast path), Safari+HTTP/1.1 as fallback for sites that block Chrome.
FETCH_STRATEGIES = [
    (CHROME_SESSION, {}),
    (SAFARI_SESSION, {"http_version": 2}),
]

_EXTRA_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%Y%m%d%H%M%S",
)


def parse_rss_date(date_str):
    """RFC 2822 (RSS), ISO 8601 (Atom / sitemaps / many CMSs) or a few common bare formats.
    Naive timestamps are taken as UTC — for a 24h window that is close enough."""
    if not date_str:
        return None
    s = date_str.strip()
    if not s:
        return None

    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    for fmt in _EXTRA_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


_TURKISH_CHAR_FIXES = str.maketrans({
    "ř": "ş", "Ř": "Ş",
    "đ": "ğ", "Đ": "Ğ",
    "¼": "ü",
    "ù": "ü", "Ù": "Ü",
    "ě": "e", "Ě": "E",
    "ő": "ş", "Ő": "Ş",
    "ţ": "ş", "Ţ": "Ş",
    "ĺ": "ş", "Ĺ": "Ş",
})

# Anti-scraping placeholder: %XYZ (3 digits) sandwiched between letters, e.g. "T%300rk".
# Some Turkish sources use a custom web font where bytes like %300 render as a real
# Turkish character. When scraped as text we just get the bytes, with no way to know
# which letter they represented — so we replace with '?' rather than guess.
# URLs use 2-hex-digit encoding (%XX), so this pattern won't false-positive on them.
_PLACEHOLDER_PATTERN = re.compile(
    r"(?<=[A-Za-zçğıöşüÇĞİÖŞÜ])%\d{3}(?=[A-Za-zçğıöşüÇĞİÖŞÜ])"
)


def normalize_turkish_chars(text):
    """Make Turkish text safe for the report. Three layers:
    1. NFC-normalize — collapses decomposed forms (r + combining caron) into ř,
       so the translation table below can actually match them.
    2. Translate known LLM/encoding substitutions (ř→ş, đ→ğ, ¼→ü, etc.).
    3. Replace unrecoverable anti-scraping placeholders (%300-style) with '?'.
    Also strips stray underscores the model sometimes inserts between Turkish chars."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TURKISH_CHAR_FIXES)
    text = re.sub(r"([şğüöçıİŞĞÜÖÇI])_(?=[a-zA-ZşğüöçıİŞĞÜÖÇI])", r"\1", text)
    return _PLACEHOLDER_PATTERN.sub("?", text)


def clean_text(text):
    if not text:
        return ""

    # Some sources double-escape entities (&amp;ccedil;). One unescape pass leaves them
    # as literal &ccedil; in the output, so loop until stable.
    for _ in range(5):
        unescaped = html_lib.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    text = text.replace("\xa0", " ")

    lines = []

    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()

        if not line:
            continue

        if any(word in lower for word in ["çerez", "cookie", "consent"]):
            continue

        if line == "Continue reading...":
            continue

        if "Get our breaking news email" in line:
            continue

        if "Listen ·" in line:
            continue

        if line.startswith("(Image credit:"):
            continue

        if any(marker in line for marker in BAD_TEXT_MARKERS):
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_bad_text(text):
    if not text:
        return True

    cleaned = clean_text(text)

    if len(cleaned) < 120:
        return True

    if any(marker in text for marker in BAD_TEXT_MARKERS) and len(cleaned) < 600:
        return True

    return False


def extract_jsonld_article_body(raw_html):
    try:
        soup = BeautifulSoup(raw_html, "lxml")
        scripts = soup.find_all("script", type="application/ld+json")

        def walk(obj):
            if isinstance(obj, dict):
                article_type = obj.get("@type")

                if isinstance(article_type, list):
                    is_article = any(
                        t in ["Article", "NewsArticle", "ReportageNewsArticle"]
                        for t in article_type
                    )
                else:
                    is_article = article_type in ["Article", "NewsArticle", "ReportageNewsArticle"]

                if is_article and obj.get("articleBody"):
                    return obj["articleBody"]

                for value in obj.values():
                    found = walk(value)
                    if found:
                        return found

            if isinstance(obj, list):
                for value in obj:
                    found = walk(value)
                    if found:
                        return found

            return ""

        for script in scripts:
            raw = script.string or script.get_text()
            if not raw:
                continue

            data = json.loads(raw)
            found = walk(data)

            if found:
                return clean_text(found)

    except Exception:
        pass

    return ""


def extract_text_from_html(raw_html, url=""):
    if not raw_html:
        return ""

    text = extract_jsonld_article_body(raw_html)
    if not is_bad_text(text):
        return text  # already cleaned inside extract_jsonld_article_body

    try:
        text = trafilatura.extract(
            raw_html,
            url=url,
            output_format="txt",
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        if not is_bad_text(text):
            return clean_text(text)

    except Exception:
        pass

    if Article:
        try:
            article = Article(url)
            article.set_html(raw_html)
            article.parse()

            if not is_bad_text(article.text):
                return clean_text(article.text)

        except Exception:
            pass

    return ""


def resolve_google_news_url(url):
    if "news.google.com/rss/articles/" not in url:
        return url

    if not gnewsdecoder:
        return url

    try:
        result = gnewsdecoder(url)

        if isinstance(result, dict):
            decoded = result.get("decoded_url") or result.get("url")
            if decoded and decoded.startswith("http"):
                return decoded

        if isinstance(result, str) and result.startswith("http"):
            return result

    except Exception:
        pass

    return url


def fetch_html(url):
    # Return raw bytes (not r.text) so downstream parsers can read the page's own meta-charset.
    # requests/curl_cffi default to ISO-8859-1 when Content-Type has no charset, which mangles
    # UTF-8 Turkish content (ş→ř, ğ→đ, etc.) consistently within each source.
    last_err = "ERROR_FETCH: no strategy attempted"

    for session, kwargs in FETCH_STRATEGIES:
        try:
            r = session.get(url, timeout=20, allow_redirects=True, **kwargs)
        except Exception as e:
            last_err = f"ERROR_FETCH: {e}"
            continue

        if r.status_code < 400:
            return r.content, r.url, ""

        if r.status_code in [401, 403, 429]:
            last_err = f"ERROR_BLOCKED: HTTP {r.status_code}"
        else:
            last_err = f"ERROR_HTTP: HTTP {r.status_code}"

    # Final fallback: plain requests (last resort, often blocked by bot detection)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        if r.status_code < 400:
            return r.content, r.url, ""
        if r.status_code in [401, 403, 429]:
            last_err = f"ERROR_BLOCKED: HTTP {r.status_code}"
        else:
            last_err = f"ERROR_HTTP: HTTP {r.status_code}"
    except Exception as e:
        last_err = f"ERROR_FETCH: {e}"

    return b"", url, last_err


def fetch_via_reader(url):
    """Fetch a page as plain text through READER_PROXY. Returns "" on any failure."""
    global _last_reader_call
    if not READER_PROXY:
        return ""
    gap = READER_MIN_INTERVAL - (time.monotonic() - _last_reader_call)
    if gap > 0:
        time.sleep(gap)
    _last_reader_call = time.monotonic()
    try:
        r = requests.get(
            READER_PROXY + url,
            headers={"Accept": "text/plain", "X-Return-Format": "text", "User-Agent": HEADERS["User-Agent"]},
            timeout=45,
        )
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        pass
    return ""


def _strip_html(fragment):
    """Feed descriptions are often HTML. Reduce to text."""
    if not fragment:
        return ""
    try:
        return BeautifulSoup(fragment, "html.parser").get_text(" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", fragment)


def _fetch_feed_bytes(feed_url):
    """Feed XML as bytes. Browser-impersonating session first (some CDNs 403 plain clients),
    then urllib as a fallback. Returns (bytes, error)."""
    try:
        r = CHROME_SESSION.get(feed_url, timeout=20, allow_redirects=True)
        if r.status_code < 400 and r.content:
            return r.content, ""
        err = f"HTTP {r.status_code}"
    except Exception as e:
        err = str(e)
    try:
        req = urllib.request.Request(feed_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read(), ""
    except Exception as e:
        return b"", f"{err}; urllib: {e}"


def _first_text(el, *names):
    for name in names:
        t = el.findtext(name)
        if t and t.strip():
            return t.strip()
    return ""


def _parse_feed_entries(root):
    """Yield raw entries {title, url, date, description} from RSS 2.0, Atom or a Google News
    sitemap. Namespaces are stripped by the caller, so tags are bare (pubDate, date, entry...).
    Returns (kind, entries)."""
    tag = root.tag.lower()
    entries = []

    if tag == "urlset":  # Google News sitemap
        for u in root.findall(".//url"):
            loc = _first_text(u, "loc")
            if not loc:
                continue
            news = u.find("news")
            title = _first_text(news, "title") if news is not None else ""
            date = (_first_text(news, "publication_date") if news is not None else "") or _first_text(u, "lastmod")
            entries.append({"title": title or loc, "url": loc, "date": date, "description": ""})
        return "sitemap", entries

    if tag == "feed":  # Atom
        for e in root.findall(".//entry"):
            url = ""
            for link in e.findall("link"):
                href = (link.get("href") or "").strip()
                if not href:
                    continue
                if link.get("rel") in (None, "", "alternate"):
                    url = href
                    break
                url = url or href
            url = url or _first_text(e, "link")
            entries.append({
                "title": _first_text(e, "title"),
                "url": url,
                "date": _first_text(e, "published", "updated", "issued", "date"),
                "description": _first_text(e, "summary", "content"),
            })
        return "atom", entries

    for item in root.findall(".//item"):  # RSS 2.0 / RSS 1.0
        entries.append({
            "title": _first_text(item, "title"),
            "url": _first_text(item, "link") or (item.find("link").get("href", "") if item.find("link") is not None else ""),
            "date": _first_text(item, "pubDate", "date", "published", "updated", "lastBuildDate"),
            "description": _first_text(item, "encoded", "description", "summary"),
        })
    return "rss", entries


def get_articles_from_rss(rss_url, hours=24, source=None):
    """Articles published in the last `hours` from an RSS/Atom feed or a Google News sitemap.
    Prints a one-line diagnosis when nothing qualifies, so a stale or oddly-dated feed is visible
    in the log instead of silently yielding 'No articles'."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    raw, err = _fetch_feed_bytes(rss_url)
    if err:
        print(f"  Failed to fetch feed: {err}")
        return []
    try:
        root = ET.fromstring(raw)
    except Exception as e:
        head = raw[:80].decode("utf-8", "replace").replace("\n", " ")
        print(f"  Failed to parse feed as XML: {e} (starts with: {head!r})")
        return []

    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    kind, entries = _parse_feed_entries(root)
    url_filter = SOURCE_URL_FILTERS.get(source)

    articles = []
    newest = None
    undated = 0
    filtered_out = 0
    sample_bad_date = None
    for entry in entries:
        title, url, date = entry["title"], entry["url"], entry["date"]
        if not title or not url:
            continue
        url = url.strip()
        if url_filter and not url_filter.search(url):
            filtered_out += 1
            continue

        dt = parse_rss_date(date)
        if dt is None:
            undated += 1
            if date and sample_bad_date is None:
                sample_bad_date = date
            continue
        if newest is None or dt > newest:
            newest = dt
        if dt < cutoff:
            continue

        articles.append({
            "title": clean_text(title),
            "url": url,
            "date": date or "no date",
            "description": _strip_html(entry.get("description", "")),
            "_dt": dt,
        })

    articles.sort(key=lambda a: a["_dt"], reverse=True)
    cap = MAX_ARTICLES_PER_SOURCE.get(source)
    if cap and len(articles) > cap:
        print(f"  {len(articles)} fresh articles, keeping the newest {cap}")
        articles = articles[:cap]
    for a in articles:
        a.pop("_dt", None)

    if not articles:
        bits = [f"{kind} feed, {len(entries)} entries"]
        if filtered_out:
            bits.append(f"{filtered_out} dropped by URL filter")
        if newest is not None:
            bits.append(f"newest dated {newest.strftime('%Y-%m-%d %H:%M')} UTC")
        if undated:
            bits.append(f"{undated} with unparseable/missing dates" + (f" (e.g. {sample_bad_date!r})" if sample_bad_date else ""))
        print("  Diagnosis: " + ", ".join(bits))

    return articles


def _feed_text_fallback(article):
    """The feed's own summary as article text, or "" if it is too short to be useful."""
    desc = (article.get("description") or "").strip()
    if len(desc) < FEED_TEXT_MIN_CHARS:
        return ""
    return FEED_TEXT_NOTE + "\n\n" + desc


def get_article_text(article):
    """(text, url, kind). kind: PAGE (fetched), READER (via reader proxy), FEED (feed summary only),
    SKIP (page fetched but no article text found), ERROR (nothing usable)."""
    original_url = article["url"]
    resolved_url = resolve_google_news_url(original_url)

    if "news.google.com/rss/articles/" in resolved_url:
        return "ERROR: Google News URL not decoded; no full text", original_url, "ERROR"

    raw_html, final_url, fetch_error = fetch_html(resolved_url)

    if not fetch_error:
        page_text = extract_text_from_html(raw_html, final_url)
        if page_text:
            return page_text, final_url, "PAGE"

    if fetch_error.startswith("ERROR_BLOCKED"):
        reader_text = fetch_via_reader(resolved_url)
        if reader_text and not is_bad_text(reader_text):
            return clean_text(reader_text), resolved_url, "READER"

    feed_text = _feed_text_fallback(article)
    if feed_text:
        return feed_text, resolved_url, "FEED"

    if fetch_error:
        return fetch_error, resolved_url, "ERROR"
    return "", final_url, "SKIP"


def collect_articles():
    """Pull articles from all configured RSS feeds. Returns dict keyed by source name."""
    by_source = defaultdict(list)
    for source, feed_url in RSS_FEEDS.items():
        print(f"=== {source} ===")
        articles = get_articles_from_rss(feed_url, hours=24, source=source)
        if not articles:
            print("  No articles found in last 24 hours.")
            continue
        breakdown = Counter()
        failures = []
        placeholder_hits = 0
        for item in articles:
            text, _, kind = get_article_text(item)
            breakdown[kind] += 1
            if kind in ("PAGE", "READER", "FEED"):
                clean_title = normalize_turkish_chars(item["title"])
                clean_body = normalize_turkish_chars(text)
                if "?" in clean_body and _PLACEHOLDER_PATTERN.search(text):
                    placeholder_hits += 1
                    print(f"    placeholder corruption from {item['url']}")
                by_source[source].append({
                    "title": clean_title,
                    "url": item["url"],
                    "text": clean_body,
                })
            elif kind == "ERROR":
                failures.append((item["title"][:60], text[:100]))
        print(f"  Collected {len(by_source[source])} of {len(articles)} (breakdown: {dict(breakdown)})")
        if placeholder_hits:
            print(f"    {placeholder_hits} article(s) had %NNN placeholder corruption (source-side obfuscation)")
        for title, err in failures[:5]:
            print(f"    fail: {err} :: {title}")
    return by_source


if __name__ == "__main__":
    articles_by_source = collect_articles()
    total = sum(len(v) for v in articles_by_source.values())
    print(f"\nFetched {total} articles across {len(articles_by_source)} sources.\n")

    for source, articles in articles_by_source.items():
        print(f"\n=== {source} ===")
        for art in articles:
            sentences = re.split(r"(?<=[.!?])\s+", art["text"].strip())
            preview = " ".join(sentences[:2]).strip()
            if len(preview) > 400:
                preview = preview[:400].rstrip() + "..."
            print(f"\n• {art['title']}")
            print(f"  {art['url']}")
            print(f"  {preview}")