"""
market_scraper.py — Async Market Intelligence Scraper
=====
Daily scheduled job that fetches current personal loan rates
from the Indian market using Tavily Search + Gemini LLM.

Architecture:
  1. Tavily searches for current loan rates
  2. Gemini extracts structured rates (low/medium/high risk)
  3. Rates are stored in PostgreSQL `market_benchmarks` table
  4. Live API queries DB for benchmarks (~2ms, never scrapes in hot path)

Usage:
    # One-time run
    python src/agents/market_scraper.py

    # Scheduled (called by APScheduler or cron)
    python src/agents/market_scraper.py --schedule
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(ROOT_DIR, '.env'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_db_session(config):
    """Create a DB session."""
    db = config['database']
    url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    return Session(), engine


# ──────────────────────────────────────────────
# Tavily Search
# ──────────────────────────────────────────────
def search_market_rates(config):
    """
    Use Tavily to search for current personal loan rates in India.
    Returns raw search results with content and URLs.
    """
    from tavily import TavilyClient

    tavily_key = os.getenv('TAVILY_API_KEY')
    if not tavily_key or tavily_key == 'your_tavily_api_key_here':
        print("⚠️  TAVILY_API_KEY not set. Using config.yaml fallback rates.")
        return None

    tavily_config = config.get('agent', {}).get('tavily', {})
    scraper_config = config.get('agent', {}).get('market_scraper', {})

    # Build search query with current date
    now = datetime.now()
    query_template = scraper_config.get(
        'search_query_template',
        'Current average personal loan interest rates in India '
        'for excellent, good, and fair credit scores {month} {year}'
    )
    query = query_template.format(month=now.strftime('%B'), year=now.year)

    print(f"🔍 Tavily Search: {query}")

    client = TavilyClient(api_key=tavily_key)
    results = client.search(
        query=query,
        search_depth=tavily_config.get('search_depth', 'advanced'),
        max_results=tavily_config.get('max_results', 5),
    )

    # Extract content from results
    search_content = []
    source_urls = []
    for result in results.get('results', []):
        search_content.append(result.get('content', ''))
        source_urls.append(result.get('url', ''))

    combined_content = '\n\n---\n\n'.join(search_content)
    print(f"   Found {len(search_content)} sources")

    return {
        'content': combined_content,
        'urls': source_urls,
        'query': query,
    }


# ──────────────────────────────────────────────
# LLM Extraction (Gemini)
# ──────────────────────────────────────────────
def extract_rates_with_llm(search_results, config):
    """
    Use Gemini to extract structured rates from Tavily search results.
    Returns a dict with low_risk_rate, medium_risk_rate, high_risk_rate.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key or google_key == 'your_gemini_api_key_here':
        print("⚠️  GOOGLE_API_KEY not set. Using config.yaml fallback rates.")
        return None

    llm_config = config.get('agent', {}).get('llm', {})

    llm = ChatGoogleGenerativeAI(
        model=llm_config.get('model', 'gemini-2.0-flash'),
        google_api_key=google_key,
        temperature=llm_config.get('temperature', 0.1),
        max_output_tokens=llm_config.get('max_tokens', 2048),
    )

    extraction_prompt = f"""You are a financial data extraction agent. Analyze the following search results 
about current personal loan interest rates in India.

Extract the average interest rates for three credit risk segments.

SEARCH RESULTS:
{search_results['content']}

INSTRUCTIONS:
1. "low_risk_rate" = average rate for borrowers with excellent credit (CIBIL 750+)
2. "medium_risk_rate" = average rate for borrowers with good credit (CIBIL 650-750)
3. "high_risk_rate" = average rate for borrowers with fair credit (CIBIL 550-650)
4. Express rates as decimals (e.g., 10.5% → 0.105)
5. If you cannot find exact rates for a segment, estimate based on the available data
6. Provide a confidence score (0.0 to 1.0) based on how much data you found

Return ONLY a valid JSON object, no other text:
{{
    "low_risk_rate": <float>,
    "medium_risk_rate": <float>,
    "high_risk_rate": <float>,
    "confidence_score": <float>,
    "notes": "<brief explanation of sources and any caveats>"
}}"""

    print("🤖 Extracting rates with Gemini...")

    response = llm.invoke(extraction_prompt)
    raw_response = response.content.strip()

    # Parse JSON from response (handle markdown code blocks)
    json_str = raw_response
    if '```json' in json_str:
        json_str = json_str.split('```json')[1].split('```')[0].strip()
    elif '```' in json_str:
        json_str = json_str.split('```')[1].split('```')[0].strip()

    try:
        extracted = json.loads(json_str)
        print(f"   ✅ Extracted rates:")
        print(f"      Low risk:    {extracted['low_risk_rate']:.3f} ({extracted['low_risk_rate']*100:.1f}%)")
        print(f"      Medium risk: {extracted['medium_risk_rate']:.3f} ({extracted['medium_risk_rate']*100:.1f}%)")
        print(f"      High risk:   {extracted['high_risk_rate']:.3f} ({extracted['high_risk_rate']*100:.1f}%)")
        print(f"      Confidence:  {extracted.get('confidence_score', 'N/A')}")
        return extracted, raw_response
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse LLM response: {e}")
        print(f"   Raw response: {raw_response[:500]}")
        return None, raw_response


# ──────────────────────────────────────────────
# Database Storage
# ──────────────────────────────────────────────
def store_benchmark(session, rates, search_results, raw_llm_response, config, fetch_type='scheduled'):
    """
    Store extracted rates in the market_benchmarks table.
    Marks previous active benchmarks as superseded.
    """
    expiry_hours = config.get('agent', {}).get('market_scraper', {}).get('benchmark_expiry_hours', 24)
    llm_model = config.get('agent', {}).get('llm', {}).get('model', 'gemini-2.0-flash')

    # Deactivate previous active benchmarks
    session.execute(
        text("UPDATE market_benchmarks SET is_active = 0 WHERE is_active = 1")
    )

    # Insert new benchmark
    session.execute(
        text("""
            INSERT INTO market_benchmarks 
            (low_risk_rate, medium_risk_rate, high_risk_rate,
             source_query, source_urls, raw_llm_response, extraction_model,
             fetch_type, is_active, confidence_score, fetched_at, expires_at)
            VALUES 
            (:low, :med, :high,
             :query, :urls, :raw_resp, :model,
             :fetch_type, 1, :confidence, :fetched, :expires)
        """),
        {
            'low': rates['low_risk_rate'],
            'med': rates['medium_risk_rate'],
            'high': rates['high_risk_rate'],
            'query': search_results.get('query', '') if search_results else '',
            'urls': ','.join(search_results.get('urls', [])) if search_results else '',
            'raw_resp': raw_llm_response or '',
            'model': llm_model,
            'fetch_type': fetch_type,
            'confidence': rates.get('confidence_score', 0.0),
            'fetched': datetime.utcnow(),
            'expires': datetime.utcnow() + timedelta(hours=expiry_hours),
        }
    )

    session.commit()
    print(f"   💾 Stored in market_benchmarks (expires in {expiry_hours}h)")


def get_fallback_rates(config):
    """
    Build rates from config.yaml when Tavily/Gemini are unavailable.
    Uses RBI-anchored base_rate + risk_premium.
    """
    benchmarks = config.get('market_benchmarks', {})
    base_rate = benchmarks.get('base_rate', 0.065)
    premiums = benchmarks.get('risk_premium', {'Low': 0.02, 'Medium': 0.045, 'High': 0.08})

    rates = {
        'low_risk_rate': base_rate + premiums.get('Low', 0.02),
        'medium_risk_rate': base_rate + premiums.get('Medium', 0.045),
        'high_risk_rate': base_rate + premiums.get('High', 0.08),
        'confidence_score': 1.0,
        'notes': 'Fallback rates from config.yaml (RBI base + risk premium)',
    }

    print(f"   📋 Using fallback rates from config.yaml:")
    print(f"      Low:    {rates['low_risk_rate']*100:.1f}%")
    print(f"      Medium: {rates['medium_risk_rate']*100:.1f}%")
    print(f"      High:   {rates['high_risk_rate']*100:.1f}%")

    return rates


# ──────────────────────────────────────────────
# Query Helper (used by the live API / agent)
# ──────────────────────────────────────────────
def get_current_benchmark(config=None):
    """
    Get today's active market benchmark from PostgreSQL.
    This is the fast path (~2ms) called by the live API.
    Falls back to config.yaml if no active benchmark exists.

    Returns:
        dict with low_risk_rate, medium_risk_rate, high_risk_rate, fetched_at
    """
    if config is None:
        config = load_config()

    session, engine = get_db_session(config)

    try:
        result = session.execute(
            text("""
                SELECT low_risk_rate, medium_risk_rate, high_risk_rate,
                       fetched_at, expires_at, confidence_score, fetch_type
                FROM market_benchmarks
                WHERE is_active = 1
                ORDER BY fetched_at DESC
                LIMIT 1
            """)
        ).fetchone()

        if result:
            benchmark = {
                'low_risk_rate': result[0],
                'medium_risk_rate': result[1],
                'high_risk_rate': result[2],
                'fetched_at': result[3].isoformat() if result[3] else None,
                'expires_at': result[4].isoformat() if result[4] else None,
                'confidence_score': result[5],
                'fetch_type': result[6],
                'source': 'database',
            }

            # Check if expired
            if result[4] and datetime.utcnow() > result[4]:
                benchmark['expired'] = True
                print(f"⚠️  Active benchmark has expired (fetched: {result[3]})")
            else:
                benchmark['expired'] = False

            return benchmark
        else:
            # No benchmark in DB — use config fallback
            fallback = get_fallback_rates(config)
            fallback['source'] = 'config_fallback'
            fallback['expired'] = False
            return fallback
    finally:
        session.close()
        engine.dispose()


# ──────────────────────────────────────────────
# Main Scraper Pipeline
# ──────────────────────────────────────────────
def run_scraper(fetch_type='scheduled'):
    """Execute the full scrape → extract → store pipeline."""
    config = load_config()

    print("=" * 60)
    print("  MARKET INTELLIGENCE SCRAPER")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Type: {fetch_type}")
    print("=" * 60)

    session, engine = get_db_session(config)

    try:
        # Step 1: Search with Tavily
        search_results = search_market_rates(config)

        raw_llm_response = None
        rates = None

        if search_results:
            # Step 2: Extract with Gemini
            rates, raw_llm_response = extract_rates_with_llm(search_results, config)

        # Step 3: Fallback if search/extraction failed
        if rates is None:
            print("\n⚠️  Tavily/Gemini unavailable. Falling back to config.yaml rates.")
            rates = get_fallback_rates(config)
            fetch_type = 'fallback'

        # Step 4: Store in PostgreSQL
        store_benchmark(session, rates, search_results, raw_llm_response, config, fetch_type)

        print("\n" + "=" * 60)
        print("  ✅ MARKET BENCHMARK UPDATED")
        print(f"  Low risk:    {rates['low_risk_rate']*100:.2f}%")
        print(f"  Medium risk: {rates['medium_risk_rate']*100:.2f}%")
        print(f"  High risk:   {rates['high_risk_rate']*100:.2f}%")
        print("=" * 60)

        return rates

    except Exception as e:
        session.rollback()
        print(f"\n❌ Scraper failed: {e}")

        # Store fallback rates so the system always has benchmarks
        try:
            fallback_rates = get_fallback_rates(config)
            store_benchmark(session, fallback_rates, None, str(e), config, 'fallback')
            print("   ✅ Fallback rates stored successfully.")
            return fallback_rates
        except Exception as e2:
            print(f"   ❌ Even fallback storage failed: {e2}")
            return None
    finally:
        session.close()
        engine.dispose()


# ──────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────
def run_scheduled():
    """Run the scraper on a daily schedule using APScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    config = load_config()
    cron_expr = config.get('agent', {}).get('market_scraper', {}).get('schedule_cron', '0 9 * * *')

    # Parse cron: "0 9 * * *" → hour=9, minute=0
    parts = cron_expr.split()
    minute, hour = int(parts[0]), int(parts[1])

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_scraper,
        'cron',
        hour=hour,
        minute=minute,
        kwargs={'fetch_type': 'scheduled'},
        id='market_rate_scraper',
        name='Daily Market Rate Scraper',
    )

    print(f"📅 Scheduler started. Will run daily at {hour:02d}:{minute:02d}")
    print("   Press Ctrl+C to stop.\n")

    # Run once immediately on startup
    print("🚀 Running initial scrape...\n")
    run_scraper(fetch_type='scheduled')

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n⏹️  Scheduler stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Market Intelligence Scraper')
    parser.add_argument('--schedule', action='store_true', help='Run on daily schedule')
    parser.add_argument('--on-demand', action='store_true', help='Run once (on-demand)')
    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_scraper(fetch_type='on_demand' if args.on_demand else 'manual')
