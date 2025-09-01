#!/usr/bin/env python3
"""
Enhanced US Podcast Analysis with Detailed Keyword Tracking
Processes all episodes from a specific show with granular keyword data for research analysis
IDENTICAL to tenet_enhanced_analysis.py but for US podcasts
"""

import os
import re
import json
import logging
import time
from datetime import datetime
from collections import defaultdict, Counter
from google.cloud import storage

# ========== CONFIG ==========
PROJECT_ID = "handy-vortex-459018-g0"
BUCKET_NAME = "podcast-dissertation-audio"
OUTPUT_DIR = "us_podcast_enhanced_results"
BATCH_SIZE = 25  # Smaller batches for more detailed processing

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('us_podcast_enhanced_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ========== GCS SETUP ==========
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# ========== ANALYSIS COMPONENTS ==========
CAUSAL_PATTERNS = [
    r"because\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"due to\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"as a result of\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"resulting in\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"so that\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"therefore\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"consequently\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"which led to\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"causing\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)"
]

COMPILED_CAUSAL_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CAUSAL_PATTERNS]

# Enhanced keyword sets with specific tracking
KEYWORD_TOPICS = {
    "identity_and_culture": [
        "men", "women", "trans", "kids", "white", "black history", "mexican",
        "black lives matter", "gender", "race", "diversity"
    ],
    "security_and_border": [
        "border", "migrants", "illegal immigrants", "law enforcement", "national guard",
        "invasion", "asylum", "deportation", "border patrol"
    ],
    "government_and_institutions": [
        "federal government", "white house", "fbi", "supreme court", "secret service",
        "government", "police", "department of justice", "congress"
    ],
    "political_figures": [
        "trump", "biden", "harris", "vance", "putin", "zelensky", "elon musk",
        "desantis", "pelosi", "mcconnell"
    ],
    "constitutional_rights": [
        "free speech", "first amendment", "second amendment", "gun rights",
        "civil liberties", "constitutional"
    ],
    "international_conflict": [
        "russia", "ukraine", "world war", "nato", "china", "taiwan",
        "middle east", "iran", "north korea"
    ],
    "conspiracy_elite": [
        "epstein", "globalists", "deep state", "mainstream narrative", "great reset",
        "agenda 2030", "new world order", "puppet masters", "shadow government",
        "world economic forum", "davos"
    ],
    "economic_concerns": [
        "inflation", "money", "tax", "economic forum", "cost of living",
        "recession", "debt", "unemployment", "wages", "housing crisis"
    ],
    "health_reproductive": [
        "mental health", "abortion", "birth control", "healthcare",
        "reproductive rights", "vaccine", "covid"
    ],
    "media_technology": [
        "washington post", "tik tok", "mainstream media", "fake news",
        "censorship", "big tech", "social media", "twitter", "facebook",
        "google", "youtube"
    ]
}

PERSUASIVE_TECHNIQUES = {
    "appeal_to_fear": [
        "threat", "danger", "they will come for you", "chaos", "destroy",
        "invasion", "crisis", "collapse", "catastrophe", "disaster"
    ],
    "emotional_language": [
        "outrage", "disgusting", "evil", "innocent", "perverted", "sinful",
        "betrayal", "corrupt", "heinous", "vile", "shocking"
    ],
    "us_vs_them": [
        "they want you to", "they are lying", "wake up", "sheeple", "we must fight",
        "real americans", "patriots", "enemies", "traitors", "establishment"
    ],
    "authority_reference": [
        "experts say", "the science is clear", "officials stated", "studies show",
        "research proves", "data shows", "according to", "sources confirm"
    ],
    "repetition_emphasis": [
        "again and again", "over and over", "time and time again",
        "repeatedly", "constantly", "continuously"
    ],
    "scapegoating": [
        "blame", "fault of", "responsible for our problems", "caused by",
        "to blame for", "the reason for"
    ],
    "patriotic_appeal": [
        "america first", "our country", "real patriots", "defend freedom",
        "american values", "founding fathers", "constitution", "liberty"
    ],
    "victimization": [
        "under attack", "being silenced", "persecution", "targeted",
        "oppressed", "censored", "cancelled"
    ]
}

# SPECIAL KEYWORD TRACKING for research questions
RESEARCH_KEYWORDS = {
    "key_political_figures": ["trump", "biden", "harris", "putin", "zelensky"],
    "countries_conflicts": ["russia", "ukraine", "china", "iran", "nato"],
    "institutions": ["fbi", "cia", "government", "congress", "white house"],
    "media_platforms": ["twitter", "facebook", "youtube", "tik tok", "google"],
    "conspiracy_terms": ["deep state", "globalists", "new world order", "great reset"]
}

# ========== ENHANCED ANALYSIS FUNCTIONS ==========
def get_all_show_files(show_name, political_orientation):
    """Get all transcript files for a specific show from GCS."""
    logger.info(f"📁 Discovering all {show_name} files...")
    prefix = f"enriched_transcripts/{political_orientation}__{show_name}__"
    blobs = bucket.list_blobs(prefix=prefix)
    show_files = [blob.name for blob in blobs if blob.name.endswith('.json')]
    logger.info(f"Found {len(show_files)} {show_name} episodes")
    return show_files

def extract_full_text(segments):
    """Extract complete text from transcript segments."""
    return ' '.join(s.get("text", "") for s in segments)

def extract_causal_relationships(text):
    """Extract causal relationships with pattern tracking."""
    results = []
    pattern_frequency = Counter()
    
    for i, compiled_pattern in enumerate(COMPILED_CAUSAL_PATTERNS):
        pattern_name = CAUSAL_PATTERNS[i].split('\\s+')[0]
        try:
            matches = compiled_pattern.finditer(text)
            for match in matches:
                results.append({
                    "cause_phrase": match.group(1).strip(),
                    "pattern_used": pattern_name,
                    "position": match.start()
                })
                pattern_frequency[pattern_name] += 1
        except Exception as e:
            logger.warning(f"Error processing pattern {pattern_name}: {e}")
    
    return results, dict(pattern_frequency)

def analyze_keywords_detailed(text, keyword_dict):
    """Enhanced keyword matching with individual keyword tracking."""
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Topic-level aggregation
    topic_hits = defaultdict(lambda: defaultdict(int))
    total_by_topic = defaultdict(int)
    
    # Individual keyword tracking
    individual_keywords = defaultdict(int)
    
    for topic, keywords in keyword_dict.items():
        topic_total = 0
        for kw in keywords:
            if ' ' in kw:  # Multi-word phrase
                count = text_lower.count(kw)
            else:  # Single word with boundaries
                count = len(re.findall(rf'\b{re.escape(kw)}\b', text_lower))
            
            if count > 0:
                topic_hits[topic][kw] = count
                individual_keywords[kw] = count  # Track individual keywords
                topic_total += count
        
        total_by_topic[topic] = topic_total
    
    # Calculate normalized frequencies
    normalized_totals = {}
    normalized_individual = {}
    if word_count > 0:
        for topic, count in total_by_topic.items():
            normalized_totals[topic] = round((count / word_count) * 1000, 2)
        for keyword, count in individual_keywords.items():
            normalized_individual[keyword] = round((count / word_count) * 1000, 2)
    
    return {
        "totals_by_topic": dict(total_by_topic),
        "normalized_per_1k_words": normalized_totals,
        "detailed_keyword_hits": dict(topic_hits),
        "individual_keywords": dict(individual_keywords),
        "individual_normalized": normalized_individual
    }

def analyze_research_keywords(text):
    """Special analysis for key research keywords."""
    text_lower = text.lower()
    word_count = len(text.split())
    
    research_keyword_counts = {}
    research_keyword_normalized = {}
    
    # Flatten research keywords and count them
    all_research_keywords = []
    for category, keywords in RESEARCH_KEYWORDS.items():
        all_research_keywords.extend(keywords)
    
    for keyword in all_research_keywords:
        if ' ' in keyword:
            count = text_lower.count(keyword)
        else:
            count = len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
        
        if count > 0:
            research_keyword_counts[keyword] = count
            if word_count > 0:
                research_keyword_normalized[keyword] = round((count / word_count) * 1000, 2)
    
    return research_keyword_counts, research_keyword_normalized

def analyze_theme_dominance(topic_totals):
    """Identify dominant themes and their relative strength."""
    if not topic_totals:
        return []
    
    total_hits = sum(topic_totals.values())
    if total_hits == 0:
        return []
    
    dominance_analysis = []
    for topic, count in sorted(topic_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = round((count / total_hits) * 100, 1)
        dominance_analysis.append({
            "topic": topic,
            "count": count,
            "percentage": percentage
        })
    
    return dominance_analysis

def calculate_influence_metrics(causal_count, persuasive_count, word_count, segment_count):
    """Calculate comprehensive influence metrics."""
    if word_count == 0:
        return {}
    
    metrics = {
        "causal_density_per_1k": round((causal_count / word_count) * 1000, 2),
        "persuasive_density_per_1k": round((persuasive_count / word_count) * 1000, 2),
        "influence_ratio": round((causal_count + persuasive_count) / word_count * 1000, 2)
    }
    
    if segment_count > 0:
        metrics.update({
            "causal_per_segment": round(causal_count / segment_count, 2),
            "persuasive_per_segment": round(persuasive_count / segment_count, 2),
            "influence_per_segment": round((causal_count + persuasive_count) / segment_count, 2)
        })
    
    return metrics

def analyze_single_episode_enhanced(file_path):
    """Enhanced analysis of a single US podcast episode with detailed keyword tracking."""
    try:
        # Load data
        blob = bucket.blob(file_path)
        content = blob.download_as_string()
        data = json.loads(content)
        
        # Extract text and basic metrics
        text = extract_full_text(data.get("segments", []))
        word_count = len(text.split())
        segment_count = len(data.get("segments", []))
        
        if word_count == 0:
            logger.warning(f"Empty episode: {file_path}")
            return None
        
        # Extract episode metadata from filename
        parts = file_path.split('__')
        if len(parts) >= 4:
            episode_name = parts[2]
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', parts[3])
            date = date_match.group(1) if date_match else 'unknown'
        else:
            episode_name = file_path
            date = 'unknown'
        
        # Perform enhanced analysis
        causal_results, pattern_freq = extract_causal_relationships(text)
        theme_analysis = analyze_keywords_detailed(text, KEYWORD_TOPICS)
        persuasive_analysis = analyze_keywords_detailed(text, PERSUASIVE_TECHNIQUES)
        research_keywords, research_normalized = analyze_research_keywords(text)
        
        # Calculate metrics
        theme_dominance = analyze_theme_dominance(theme_analysis["totals_by_topic"])
        persuasive_dominance = analyze_theme_dominance(persuasive_analysis["totals_by_topic"])
        
        total_persuasive_hits = sum(persuasive_analysis["totals_by_topic"].values())
        influence_metrics = calculate_influence_metrics(
            len(causal_results), total_persuasive_hits, word_count, segment_count
        )
        
        # Compile enhanced result
        result = {
            "metadata": {
                "title": episode_name,
                "source_file": file_path,
                "date": date,  # Add date to metadata
                "segment_count": segment_count,
                "total_word_count": word_count,
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "causal_analysis": {
                "total_relationships": len(causal_results),
                "pattern_frequency": pattern_freq,
                "density_per_1k_words": influence_metrics.get("causal_density_per_1k", 0),
                "sample_relationships": causal_results[:3]  # Store samples
            },
            "thematic_analysis": {
                "totals_by_topic": theme_analysis["totals_by_topic"],
                "normalized_frequencies": theme_analysis["normalized_per_1k_words"],
                "detailed_keyword_hits": theme_analysis["detailed_keyword_hits"],
                "individual_keywords": theme_analysis["individual_keywords"],
                "individual_normalized": theme_analysis["individual_normalized"],
                "theme_dominance_ranking": theme_dominance
            },
            "persuasive_analysis": {
                "totals_by_technique": persuasive_analysis["totals_by_topic"],
                "normalized_frequencies": persuasive_analysis["normalized_per_1k_words"],
                "detailed_technique_hits": persuasive_analysis["detailed_keyword_hits"],
                "technique_dominance_ranking": persuasive_dominance
            },
            "research_keywords": {
                "raw_counts": research_keywords,
                "normalized_per_1k": research_normalized
            },
            "influence_metrics": influence_metrics,
            "narrative_fingerprint": {
                "dominant_themes": [t["topic"] for t in theme_dominance[:3]],
                "dominant_techniques": [t["topic"] for t in persuasive_dominance[:3]],
                "influence_intensity": influence_metrics.get("influence_ratio", 0),
                "content_density": influence_metrics.get("influence_per_segment", 0)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze {file_path}: {e}")
        return None

def save_enhanced_batch_results(results, batch_num, show_name, political_orientation):
    """Save enhanced batch results locally and to GCS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orientation_clean = political_orientation.replace(' ', '_')
    show_name_clean = show_name.replace(' ', '_')
    filename = f"{orientation_clean}_{show_name_clean}_enhanced_batch_{batch_num:03d}_{timestamp}.json"
    
    # Save locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(local_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Upload to GCS
    gcs_path = f"us_podcast_enhanced_analysis/{filename}"
    try:
        output_blob = bucket.blob(gcs_path)
        output_blob.upload_from_filename(local_path)
        logger.info(f"☁️ Uploaded enhanced batch {batch_num} to gs://{BUCKET_NAME}/{gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload enhanced batch {batch_num}: {e}")
    
    return local_path, gcs_path

def create_enhanced_summary(all_results):
    """Create enhanced summary statistics with detailed keyword analysis."""
    total_episodes = len(all_results)
    
    # Aggregate all research keywords
    all_research_keyword_counts = Counter()
    episode_keyword_appearances = defaultdict(int)
    
    for result in all_results:
        for keyword, count in result["research_keywords"]["raw_counts"].items():
            all_research_keyword_counts[keyword] += count
            episode_keyword_appearances[keyword] += 1
    
    # Calculate keyword statistics
    keyword_stats = {}
    for keyword, total_count in all_research_keyword_counts.items():
        episodes_with_keyword = episode_keyword_appearances[keyword]
        keyword_stats[keyword] = {
            "total_mentions": total_count,
            "episodes_mentioned": episodes_with_keyword,
            "percentage_of_episodes": round((episodes_with_keyword / total_episodes) * 100, 1),
            "avg_mentions_per_episode": round(total_count / episodes_with_keyword, 2) if episodes_with_keyword > 0 else 0
        }
    
    # Basic aggregations (from previous summary)
    total_words = sum(r["metadata"]["total_word_count"] for r in all_results)
    total_causal = sum(r["causal_analysis"]["total_relationships"] for r in all_results)
    total_persuasive = sum(sum(r["persuasive_analysis"]["totals_by_technique"].values()) for r in all_results)
    
    # Temporal analysis
    temporal_analysis = {}
    episodes_by_month = defaultdict(list)
    
    for result in all_results:
        date_str = result["metadata"].get("date", "unknown")
        if date_str != "unknown":
            # Extract year-month for grouping
            year_month = date_str[:7]  # "2024-03"
            episodes_by_month[year_month].append(result)
    
    # Calculate monthly averages
    for year_month, episodes in sorted(episodes_by_month.items()):
        monthly_influence = [ep["influence_metrics"]["influence_ratio"] for ep in episodes]
        monthly_causal = [ep["influence_metrics"]["causal_density_per_1k"] for ep in episodes]
        monthly_persuasive = [ep["influence_metrics"]["persuasive_density_per_1k"] for ep in episodes]
        
        temporal_analysis[year_month] = {
            "episode_count": len(episodes),
            "avg_influence_intensity": round(sum(monthly_influence) / len(monthly_influence), 2),
            "avg_causal_density": round(sum(monthly_causal) / len(monthly_causal), 2),
            "avg_persuasive_density": round(sum(monthly_persuasive) / len(monthly_persuasive), 2),
            "similarity_to_tenet": round(100 - abs((sum(monthly_influence) / len(monthly_influence) - 5.45) / 5.45 * 100), 1)
        }
    
    enhanced_summary = {
        "dataset_overview": {
            "total_episodes": total_episodes,
            "total_words": total_words,
            "avg_words_per_episode": round(total_words / total_episodes, 1)
        },
        "influence_metrics_summary": {
            "total_causal_relationships": total_causal,
            "total_persuasive_instances": total_persuasive,
            "avg_influence_intensity": round(sum(r["influence_metrics"]["influence_ratio"] for r in all_results) / total_episodes, 2)
        },
        "temporal_analysis": temporal_analysis,
        "research_keyword_analysis": keyword_stats,
        "top_keywords_by_mentions": dict(all_research_keyword_counts.most_common(10)),
        "analysis_timestamp": datetime.utcnow().isoformat()
    }
    
    return enhanced_summary

def main():
    """Main enhanced processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced US podcast analysis')
    parser.add_argument('--show', type=str, required=True,
                       help='Show name to analyze (e.g., thejoeroganexperience)')
    parser.add_argument('--orientation', type=str, required=True,
                       choices=['Left wing', 'Right wing'],
                       help='Political orientation')
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info(f"🚀 Starting enhanced {args.show} analysis with detailed keyword tracking...")
    
    # Get all show files
    show_files = get_all_show_files(args.show, args.orientation)
    if not show_files:
        logger.error(f"No {args.show} files found!")
        return
    
    logger.info(f"Processing {len(show_files)} {args.show} episodes in batches of {BATCH_SIZE}")
    logger.info("📊 Enhanced analysis includes detailed keyword tracking for:")
    logger.info("   • Individual keyword counts")
    logger.info("   • Research-specific keywords (Trump, Russia, etc.)")
    logger.info("   • Detailed thematic breakdowns")
    
    all_results = []
    failed_files = []
    
    # Process in smaller batches for enhanced analysis
    for i in range(0, len(show_files), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch_files = show_files[i:i + BATCH_SIZE]
        
        logger.info(f"📊 Processing enhanced batch {batch_num}/{(len(show_files) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_files)} files)")
        
        batch_results = []
        for file_path in batch_files:
            result = analyze_single_episode_enhanced(file_path)
            if result:
                batch_results.append(result)
                all_results.append(result)
            else:
                failed_files.append(file_path)
        
        # Save enhanced batch results
        if batch_results:
            save_enhanced_batch_results(batch_results, batch_num, args.show, args.orientation)
            logger.info(f"✅ Enhanced batch {batch_num} complete: {len(batch_results)} episodes processed")
            
            # Calculate and display temporal trends
            batch_dates = []
            batch_influence = []
            for result in batch_results:
                # Extract date from metadata
                date_str = result["metadata"].get("date", "unknown")
                if date_str != "unknown":
                    batch_dates.append(date_str)
                    batch_influence.append(result["influence_metrics"]["influence_ratio"])
            
            if batch_influence:
                avg_batch_influence = sum(batch_influence) / len(batch_influence)
                tenet_baseline = 5.45  # From your Tenet analysis
                similarity_to_tenet = 100 - abs((avg_batch_influence - tenet_baseline) / tenet_baseline * 100)
                
                logger.info(f"📊 Batch temporal analysis:")
                logger.info(f"   Date range: {min(batch_dates)} to {max(batch_dates)}")
                logger.info(f"   Avg influence intensity: {avg_batch_influence:.2f}")
                logger.info(f"   Similarity to Tenet baseline: {similarity_to_tenet:.1f}%")
                logger.info(f"   Difference from Tenet: {'+' if avg_batch_influence > tenet_baseline else ''}{avg_batch_influence - tenet_baseline:.2f}")
        
        # Brief pause between batches
        time.sleep(2)
    
    # Create enhanced summary
    logger.info("📊 Creating enhanced summary with keyword analysis...")
    enhanced_summary = create_enhanced_summary(all_results)
    
    # Add show metadata to summary
    enhanced_summary["show_metadata"] = {
        "show_name": args.show,
        "political_orientation": args.orientation
    }
    
    orientation_clean = args.orientation.replace(' ', '_')
    show_name_clean = args.show.replace(' ', '_')
    summary_file = f"{orientation_clean}_{show_name_clean}_enhanced_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(enhanced_summary, f, indent=2)
    
    # Upload summary to GCS
    gcs_summary_path = f"us_podcast_enhanced_analysis/{summary_file}"
    try:
        summary_blob = bucket.blob(gcs_summary_path)
        summary_blob.upload_from_filename(summary_file)
        logger.info(f"☁️ Enhanced summary uploaded to gs://{BUCKET_NAME}/{gcs_summary_path}")
    except Exception as e:
        logger.error(f"Failed to upload enhanced summary: {e}")
    
    # Final report
    end_time = time.time()
    duration = round((end_time - start_time) / 60, 1)
    
    logger.info("🎉 ENHANCED BATCH PROCESSING COMPLETE!")
    logger.info(f"⏱️  Duration: {duration} minutes")
    logger.info(f"✅ Successfully processed: {len(all_results)} episodes")
    logger.info(f"❌ Failed: {len(failed_files)} episodes")
    
    # Show top research keywords
    top_keywords = enhanced_summary["top_keywords_by_mentions"]
    logger.info("🔍 Top research keywords found:")
    for keyword, count in list(top_keywords.items())[:5]:
        episodes_mentioned = enhanced_summary["research_keyword_analysis"][keyword]["episodes_mentioned"]
        percentage = enhanced_summary["research_keyword_analysis"][keyword]["percentage_of_episodes"]
        logger.info(f"   • {keyword}: {count} total mentions across {episodes_mentioned} episodes ({percentage}%)")
    
    # Show temporal trends
    if "temporal_analysis" in enhanced_summary and enhanced_summary["temporal_analysis"]:
        logger.info("\n📈 Temporal Analysis - Similarity to Tenet over time:")
        for month, data in sorted(enhanced_summary["temporal_analysis"].items())[-6:]:  # Last 6 months
            logger.info(f"   {month}: {data['avg_influence_intensity']:.2f} intensity, "
                       f"{data['similarity_to_tenet']:.1f}% similar to Tenet (n={data['episode_count']})")
        
        # Check election period (Oct-Nov 2024)
        election_months = ["2024-10", "2024-11"]
        election_data = [enhanced_summary["temporal_analysis"].get(m) for m in election_months if m in enhanced_summary["temporal_analysis"]]
        if election_data:
            avg_election_similarity = sum(d["similarity_to_tenet"] for d in election_data) / len(election_data)
            logger.info(f"\n🗳️ Election period average similarity to Tenet: {avg_election_similarity:.1f}%")
    
    if failed_files:
        logger.warning(f"Failed files: {failed_files[:3]}...")
    
    return enhanced_summary

if __name__ == "__main__":
    summary = main()