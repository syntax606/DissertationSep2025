#!/usr/bin/env python3
“””
Streamlined 5-Feature NLP Enrichment Pipeline for Podcast Segments
Features: Sentence-BERT, Granular Emotions, Persuasive Techniques, Basic Sentiment, Topic Modeling
“””

import json
import os
import glob
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re

# Persuasive technique keywords (simple but effective)

PERSUASIVE_TECHNIQUES = {
‘loaded_language’: [
‘radical’, ‘extreme’, ‘dangerous’, ‘corrupt’, ‘elite’, ‘establishment’,
‘mainstream media’, ‘fake news’, ‘propaganda’, ‘brainwashed’, ‘sheep’,
‘deep state’, ‘globalist’, ‘agenda’, ‘narrative’, ‘indoctrination’
],
‘fear_appeal’: [
‘threat’, ‘danger’, ‘crisis’, ‘emergency’, ‘catastrophe’, ‘disaster’,
‘collapse’, ‘destroy’, ‘attack’, ‘invasion’, ‘war’, ‘violence’,
‘afraid’, ‘scared’, ‘terrifying’, ‘nightmare’
],
‘whataboutism’: [
‘what about’, ‘but what about’, ‘meanwhile’, ‘while’, ‘instead of’,
‘rather than’, ‘compared to’, ‘unlike’, ‘whereas’
],
‘causal_oversimplification’: [
‘because of’, ‘thanks to’, ‘due to’, ‘caused by’, ‘result of’,
‘blame’, ‘fault’, ‘responsible for’, ‘leads to’, ‘creates’
]
}

# Topic categories for keyword-based detection

PREDEFINED_TOPICS = {
‘politics’: [‘election’, ‘president’, ‘government’, ‘policy’, ‘democrat’, ‘republican’,
‘congress’, ‘senate’, ‘vote’, ‘campaign’, ‘politician’, ‘administration’,
‘biden’, ‘trump’, ‘harris’, ‘political’, ‘legislation’, ‘white house’],
‘conspiracy’: [‘deep state’, ‘conspiracy’, ‘cover up’, ‘elite’, ‘agenda’, ‘narrative’,
‘mainstream media’, ‘globalist’, ‘shadow’, ‘secret’, ‘hidden’, ‘truth’,
‘psyop’, ‘false flag’, ‘controlled’, ‘puppet’, ‘awakening’, ‘red pill’],
‘health’: [‘vaccine’, ‘covid’, ‘health’, ‘medical’, ‘doctor’, ‘treatment’, ‘pandemic’,
‘virus’, ‘immunity’, ‘pharmaceutical’, ‘medicine’, ‘hospital’, ‘pfizer’,
‘moderna’, ‘mrna’, ‘side effects’, ‘natural immunity’, ‘ivermectin’],
‘economy’: [‘economy’, ‘inflation’, ‘market’, ‘money’, ‘dollar’, ‘bank’, ‘financial’,
‘recession’, ‘jobs’, ‘unemployment’, ‘stock’, ‘crypto’, ‘currency’,
‘federal reserve’, ‘interest rates’, ‘housing’, ‘debt’, ‘collapse’],
‘technology’: [‘ai’, ‘tech’, ‘social media’, ‘algorithm’, ‘data’, ‘privacy’, ‘internet’,
‘digital’, ‘surveillance’, ‘platform’, ‘silicon valley’, ‘censorship’,
‘twitter’, ‘facebook’, ‘google’, ‘tiktok’, ‘artificial intelligence’],
‘conflict’: [‘war’, ‘military’, ‘ukraine’, ‘russia’, ‘china’, ‘nuclear’, ‘weapon’,
‘defense’, ‘nato’, ‘conflict’, ‘invasion’, ‘soldier’, ‘pentagon’,
‘drone’, ‘missile’, ‘border’, ‘israel’, ‘gaza’, ‘middle east’],
‘culture’: [‘woke’, ‘culture’, ‘gender’, ‘trans’, ‘identity’, ‘cancel’, ‘diversity’,
‘traditional’, ‘values’, ‘society’, ‘education’, ‘children’, ‘pronoun’,
‘critical race’, ‘grooming’, ‘indoctrination’, ‘family’, ‘christian’],
‘media’: [‘media’, ‘news’, ‘journalist’, ‘report’, ‘propaganda’, ‘narrative’, ‘story’,
‘coverage’, ‘bias’, ‘fact check’, ‘misinformation’, ‘disinformation’,
‘alternative media’, ‘independent’, ‘mainstream’, ‘legacy media’]
}

def load_models():
“”“Load all required models once at startup.”””
print(“🔄 Loading models…”)

```
models = {
    'sentence_bert': SentenceTransformer('all-mpnet-base-v2'),
    'emotion': pipeline('text-classification', 
                      'j-hartmann/emotion-english-distilroberta-base',
                      return_all_scores=True),
    'sentiment': pipeline('sentiment-analysis', 
                        'distilbert-base-uncased-finetuned-sst-2-english')
}

print("✅ All models loaded successfully!")
return models
```

def detect_persuasive_techniques(text):
“”“Detect persuasive techniques using keyword matching.”””
text_lower = text.lower()
detected = []

```
for technique, keywords in PERSUASIVE_TECHNIQUES.items():
    for keyword in keywords:
        if keyword in text_lower:
            detected.append(technique)
            break  # Only count each technique once per segment

return detected
```

def detect_topics_keywords(text):
“”“Keyword-based topic detection with confidence scores.”””
text_lower = text.lower()
detected_topics = []
topic_scores = {}

```
for topic, keywords in PREDEFINED_TOPICS.items():
    matches = sum(1 for keyword in keywords if keyword in text_lower)
    if matches > 0:
        # Score based on number of keyword matches normalized by total keywords
        score = matches / len(keywords)
        topic_scores[topic] = round(score, 3)
        detected_topics.append(topic)

return detected_topics, topic_scores
```

def extract_key_phrases(text, max_features=5):
“”“Extract key phrases using TF-IDF.”””
try:
# Use TF-IDF to find important n-grams
vectorizer = TfidfVectorizer(
max_features=max_features,
ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
stop_words=‘english’,
min_df=1
)

```
    # Fit and transform the single document
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the TF-IDF scores
    scores = tfidf_matrix.toarray()[0]
    
    # Create phrase-score pairs and sort by score
    phrase_scores = [(feature_names[i], scores[i]) 
                    for i in range(len(scores)) if scores[i] > 0]
    phrase_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top phrases
    return [phrase for phrase, score in phrase_scores[:max_features]]
except Exception as e:
    return []
```

def get_emotion_scores(emotion_results):
“”“Convert emotion pipeline results to clean scores.”””
emotion_scores = {}

```
for result in emotion_results[0]:  # First (and only) result
    emotion_scores[result['label']] = round(result['score'], 3)

return emotion_scores
```

def enrich_segment(segment, models):
“”“Enrich a single segment with all 5 features.”””
text = segment.get(‘text’, ‘’)

```
# Skip very short segments
if len(text.strip()) < 10:
    return segment

try:
    # 1. Sentence-BERT Embeddings
    embedding = models['sentence_bert'].encode(text)
    segment['embedding'] = embedding.tolist()  # Convert numpy to list for JSON
    
    # 2. Granular Emotion Classification
    emotion_results = models['emotion'](text)
    emotion_scores = get_emotion_scores(emotion_results)
    segment['emotion_scores'] = emotion_scores
    
    # Get dominant emotion
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    segment['dominant_emotion'] = {
        'label': dominant_emotion[0],
        'score': dominant_emotion[1]
    }
    
    # 3. Basic Sentiment
    sentiment_result = models['sentiment'](text)[0]
    segment['sentiment'] = {
        'label': sentiment_result['label'],
        'score': round(sentiment_result['score'], 3)
    }
    
    # 4. Persuasive Techniques
    techniques = detect_persuasive_techniques(text)
    segment['persuasive_techniques'] = techniques
    
    # 5. Topic Modeling
    topics, topic_scores = detect_topics_keywords(text)
    segment['detected_topics'] = topics
    segment['topic_scores'] = topic_scores
    
    # Key phrase extraction
    key_phrases = extract_key_phrases(text)
    segment['key_phrases'] = key_phrases
    
    # Primary topic (highest scoring)
    if topic_scores:
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])
        segment['primary_topic'] = {
            'topic': primary_topic[0],
            'confidence': primary_topic[1]
        }
    else:
        segment['primary_topic'] = None
    
    # Add processing metadata
    segment['enriched_at'] = datetime.now().isoformat()
    segment['enrichment_version'] = '1.1_with_topics'
    
except Exception as e:
    print(f"⚠️ Error enriching segment: {e}")
    # Add empty enrichment fields to maintain consistency
    segment['embedding'] = None
    segment['emotion_scores'] = {}
    segment['dominant_emotion'] = {'label': 'unknown', 'score': 0.0}
    segment['sentiment'] = {'label': 'UNKNOWN', 'score': 0.0}
    segment['persuasive_techniques'] = []
    segment['detected_topics'] = []
    segment['topic_scores'] = {}
    segment['key_phrases'] = []
    segment['primary_topic'] = None

return segment
```

def aggregate_episode_topics(segments):
“”“Aggregate topics across all segments in an episode.”””
all_topics = {}
total_segments = len(segments)

```
for segment in segments:
    for topic in segment.get('detected_topics', []):
        all_topics[topic] = all_topics.get(topic, 0) + 1

# Calculate topic percentages
topic_distribution = {
    topic: {
        'count': count,
        'percentage': round(count / total_segments * 100, 2)
    }
    for topic, count in all_topics.items()
}

# Sort by count
sorted_topics = sorted(topic_distribution.items(), 
                      key=lambda x: x[1]['count'], 
                      reverse=True)

return dict(sorted_topics)
```

def process_file(file_path, models, output_dir):
“”“Process a single JSON file.”””
try:
with open(file_path, ‘r’, encoding=‘utf-8’) as f:
data = json.load(f)
except Exception as e:
print(f”❌ Error reading {file_path}: {e}”)
return 0, 0

```
segments = data.get('segments', [])
if not segments:
    print(f"⚠️ No segments in {file_path}")
    return 0, 0

processed = 0
skipped = 0

# Process segments with progress bar
for segment in tqdm(segments, desc=f"Processing {os.path.basename(file_path)}", leave=False):
    try:
        enrich_segment(segment, models)
        processed += 1
    except Exception as e:
        print(f"⚠️ Skipping segment: {e}")
        skipped += 1

# Add episode-level topic summary
data['episode_topic_summary'] = aggregate_episode_topics(segments)

# Save enriched file
output_file = output_dir / f"{Path(file_path).stem}_enriched.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

return processed, skipped
```

def main():
parser = argparse.ArgumentParser(description=‘Streamlined NLP enrichment for podcast segments’)
parser.add_argument(‘input_dir’, help=‘Directory containing processed transcript files’)
parser.add_argument(’–output’, ‘-o’, required=True, help=‘Output directory for enriched files’)
parser.add_argument(’–batch-size’, type=int, default=50, help=‘Number of files to process in batch’)

```
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

# Find all JSON files
json_files = list(input_dir.glob('*_with_ids.json'))

if not json_files:
    print(f"❌ No JSON files found in {input_dir}")
    return

print(f"📂 Found {len(json_files)} files to enrich")
print(f"💾 Output directory: {output_dir}")

# Load models once
models = load_models()

total_processed = 0
total_skipped = 0

# Process files
for file_path in tqdm(json_files, desc="Processing files"):
    processed, skipped = process_file(file_path, models, output_dir)
    total_processed += processed
    total_skipped += skipped
    
    # Log progress every 100 files
    if (total_processed + total_skipped) % 1000 == 0:
        print(f"📊 Progress: {total_processed:,} segments enriched, {total_skipped:,} skipped")

print("=" * 60)
print(f"🎉 Enrichment complete!")
print(f"✅ Total segments enriched: {total_processed:,}")
print(f"⚠️ Total segments skipped: {total_skipped:,}")
print(f"📁 Enriched files saved to: {output_dir}")
print(f"🏷️ Features added: embeddings, emotions, sentiment, persuasive techniques, topics")
```

if **name** == “**main**”:
main()

