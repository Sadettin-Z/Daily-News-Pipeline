"""Diagnostic: fetch every configured feed, report what it contains, and fetch one article
per source. Makes no Gemini calls. Run locally or via the 'Feed Check' workflow."""
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import scraper


def main(hours=24):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    problems = []
    for source, url in scraper.RSS_FEEDS.items():
        print(f"=== {source} ===\n  {url}")
        raw, err = scraper._fetch_feed_bytes(url)
        if err:
            print(f"  FEED FETCH FAILED: {err}")
            problems.append(source)
            continue
        print(f"  {len(raw)} bytes, starts: {raw[:100].decode('utf-8', 'replace')!r}")
        try:
            root = ET.fromstring(raw)
        except Exception as e:
            print(f"  NOT XML: {e}")
            problems.append(source)
            continue
        for el in root.iter():
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        kind, entries = scraper._parse_feed_entries(root)
        dates = [(e["date"], scraper.parse_rss_date(e["date"])) for e in entries]
        parsed = [d for _, d in dates if d]
        fresh = [d for d in parsed if d >= cutoff]
        print(f"  {kind}: {len(entries)} entries, {len(parsed)} dated, {len(fresh)} within {hours}h")
        if parsed:
            print(f"  newest: {max(parsed).isoformat()}  oldest: {min(parsed).isoformat()}")
        bad = [raw_d for raw_d, d in dates if raw_d and not d]
        if bad:
            print(f"  unparseable dates ({len(bad)}), e.g. {bad[:3]}")
        if entries:
            e = entries[0]
            print(f"  first entry: title={e['title'][:70]!r} url={e['url'][:90]!r} date={e['date']!r} desc_len={len(e.get('description') or '')}")
        arts = scraper.get_articles_from_rss(url, hours=hours, source=source)
        if not arts:
            problems.append(source)
            continue
        a = arts[0]
        text, final_url, kind = scraper.get_article_text(a)
        print(f"  article fetch [{kind}] {final_url[:90]} -> {len(text)} chars"
              + (f": {text[:120]!r}" if kind != "ERROR" else f": {text}"))
        if kind == "ERROR":
            problems.append(source)
    print("\nSources with problems:", ", ".join(problems) or "none")


def probe(urls):
    """Fetch arbitrary URLs and show status, content type and the first bytes. Handy for
    hunting a replacement feed (try /robots.txt first: it lists sitemaps)."""
    for url in urls:
        print(f"--- {url}")
        try:
            r = scraper.CHROME_SESSION.get(url, timeout=20, allow_redirects=True)
            ctype = r.headers.get("content-type", "?")
            print(f"  HTTP {r.status_code} {ctype} {len(r.content)} bytes final={r.url}")
            body = r.content[:1500].decode("utf-8", "replace")
            for line in body.splitlines()[:25]:
                print(f"  | {line[:160]}")
        except Exception as e:
            print(f"  ERROR {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe(sys.argv[2:])
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
