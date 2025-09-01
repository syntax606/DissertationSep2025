#!/usr/bin/env python3
"""
Complete Political Discourse Analyzer - Enhanced with Detailed Reports (Fixed)
Includes: Orientation-specific tracking, normalization, show-level analysis, temporal patterns,
and comprehensive reporting matching timeline script detail.

Fixes:
- Correct indentation for report methods and helpers.
- Compute `show_intensity` inside save_complete_results() to avoid NameError.
- Track show-level persuasive technique counts (was incorrectly aggregated at orientation level).
- Ensure OUTPUT_DIR exists in all report generators.
- Safer, case-tolerant research-entity lookups in JSON summary.
"""

import json
import re
import os
from collections import defaultdict, Counter
from google.cloud import storage
import logging
from datetime import datetime
import pandas as pd

# Configuration
PROJECT_ID = "handy-vortex-459018-g0"
BUCKET_NAME = "podcast-dissertation-audio"
OUTPUT_DIR = "political_discourse_complete_analysis"
CONTEXT_WINDOW = 300

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UNIFIED POLITICAL KEYWORDS
CASE_INSENSITIVE_ENTITIES = {
    "figures": ['trump', 'biden', 'harris', 'obama', 'clinton', 'pelosi', 'mcconnell',
                'pence', 'schumer', 'sanders', 'desantis', 'putin', 'zelensky',
                'zelenskyy', 'vance', 'elon musk'],
    "parties": ['democrats', 'republicans', 'liberals', 'conservatives', 'the left',
                'the right', 'dems'],
    "institutions": ['government', 'congress', 'white house', 'supreme court',
                     'senate', 'house', 'kremlin', 'pentagon', 'federal government',
                     'department of justice', 'secret service', 'national guard'],
    "media": ['media', 'mainstream media', 'fake news', 'big tech', 'censorship'],
    "groups": ['establishment', 'elites', 'deep state', 'swamp', 'globalists',
               'new world order', 'great reset', 'puppet masters', 'shadow government',
               'world economic forum', 'davos', 'mainstream narrative', 'agenda 2030'],
    "countries": ['russia', 'ukraine', 'china', 'israel', 'iran', 'america',
                  'united states', 'north korea', 'taiwan'],
    "identity": ['men', 'women', 'trans', 'kids', 'white', 'black', 'gender', 'race',
                 'diversity', 'black lives matter', 'black history', 'mexican'],
    "border": ['border', 'migrants', 'illegal immigrants', 'invasion', 'asylum',
               'deportation', 'border patrol', 'law enforcement'],
    "economy": ['inflation', 'money', 'tax', 'economic forum', 'cost of living',
                'recession', 'debt', 'unemployment', 'wages', 'housing crisis'],
    "health": ['mental health', 'abortion', 'birth control', 'healthcare',
               'reproductive rights', 'vaccine', 'covid'],
    "rights": ['free speech', 'first amendment', 'second amendment', 'gun rights',
               'civil liberties', 'constitutional'],
    "social_media": ['twitter', 'facebook', 'youtube', 'tik tok', 'google',
                     'social media'],
    "conspiracy": ['epstein']
}

CASE_SENSITIVE_ENTITIES = {
    "parties": ['GOP', 'MAGA', 'DNC'],
    "institutions": ['FBI', 'CIA', 'DOJ', 'NATO', 'EU'],
    "media": ['CNN', 'FOX', 'MSNBC', 'NYT', 'WAPO', 'RT'],
    "organizations": ['UN', 'WHO'],
    "countries": ['US', 'USA', 'UK']
}

SPECIAL_ENTITIES = {
    "figures": [('AOC', r'\bAOC\b'), ('aoc', r'\baoc\b')]
}

# Blame patterns
BLAME_PATTERNS = [
    r'\b(?:is|are|was|were)\s+(?:to\s+)?blame\b',
    r'\bresponsible\s+for\b',
    r'\b(?:it\'s|its|that\'s)\s+\w+\'s\s+fault\b',
    r'\bcaused\s+by\b',
    r'\bdue\s+to\b',
    r'\bbecause\s+of\b',
    r'\bblame\s+(?:it\s+)?on\b',
    r'\bscapegoat',
    r'\bthey\s+blame\b',
    r'\beverything\s+is\s+\w+\'s\s+fault\b',
    r'\bpointing\s+fingers?\b',
    r'\bblame\s+game\b',
    r'\bfault of\b',
    r'\bresponsible for our problems\b',
    r'\bto blame for\b',
    r'\bthe reason for\b'
]

# Persuasive language patterns
PERSUASIVE_PATTERNS = {
    "emotional_appeals": {
        "fear": [r'\b(?:terrifying|frightening|scary|horrifying|alarming)\b',
                 r'\b(?:danger|dangerous|threat|threatening|peril)\b',
                 r'\b(?:destroy|destroying|devastate|catastrophe)\b',
                 r'\bthey will come for you\b', r'\bchaos\b', r'\bcollapse\b',
                 r'\binvasion\b', r'\bcrisis\b', r'\bdisaster\b'],
        "anger": [r'\b(?:outrageous|infuriating|disgusting|despicable)\b',
                  r'\b(?:corrupt|corruption|crooked|evil)\b',
                  r'\bbetrayal\b', r'\bheinous\b', r'\bvile\b', r'\bshocking\b'],
        "disgust": [r'\b(?:sick|sickening|vile|repulsive|revolting)\b',
                    r'\bperverted\b', r'\bsinful\b'],
        "pride": [r'\b(?:patriot|patriotic|american values|freedom)\b',
                  r'\breal patriots\b', r'\bdefend freedom\b', r'\bfounding fathers\b']
    },
    "hyperbole": {
        "absolute": [r'\b(?:always|never|every|all|none|nothing|everything)\b',
                     r'\b(?:completely|totally|absolutely|entirely)\b'],
        "extreme": [r'\b(?:disaster|catastrophe|apocalypse|doomsday)\b',
                    r'\b(?:miracle|perfect|flawless|terrible|horrible)\b']
    },
    "loaded_language": {
        "political": [r'\b(?:radical|extremist|far-left|far-right|socialist|fascist)\b',
                      r'\b(?:regime|mob|elites|establishment)\b',
                      r'\b(?:fake news|hoax|witch hunt|deep state)\b'],
        "moral": [r'\b(?:un-american|anti-american|traitor|patriot)\b',
                  r'\b(?:immoral|evil|righteous|virtue)\b',
                  r'\binnocent\b']
    },
    "certainty_markers": {
        "epistemic": [r'\b(?:obviously|clearly|definitely|undoubtedly)\b',
                      r'\b(?:of course|no question|without a doubt)\b',
                      r'\b(?:fact is|truth is|reality is)\b',
                      r'\bthe science is clear\b', r'\bexperts say\b',
                      r'\bstudies show\b', r'\bresearch proves\b'],
        "hedging": [r'\b(?:maybe|perhaps|possibly|might|could)\b',
                    r'\b(?:seems|appears|allegedly|supposedly)\b']
    },
    "us_vs_them": {
        "ingroup": [r'\b(?:we|us|our)\b.*\b(?:they|them|their)\b',
                    r'\b(?:real americans|true patriots|honest people)\b',
                    r'\bour country\b', r'\bamerica first\b'],
        "outgroup": [r'\b(?:those people|these people|they)\b',
                     r'\b(?:the other side|opponents|enemies)\b',
                     r'\bthey want you to\b', r'\bthey are lying\b',
                     r'\bsheeple\b', r'\bwe must fight\b']
    },
    "authority_reference": {
        "appeal": [r'\bexperts say\b', r'\bthe science is clear\b',
                   r'\bofficials stated\b', r'\bstudies show\b',
                   r'\bresearch proves\b', r'\bdata shows\b',
                   r'\baccording to\b', r'\bsources confirm\b']
    },
    "repetition_emphasis": {
        "patterns": [r'\bagain and again\b', r'\bover and over\b',
                     r'\btime and time again\b', r'\brepeatedly\b',
                     r'\bconstantly\b', r'\bcontinuously\b']
    },
    "victimization": {
        "patterns": [r'\bunder attack\b', r'\bbeing silenced\b',
                     r'\bpersecution\b', r'\btargeted\b', r'\boppressed\b',
                     r'\bcensored\b', r'\bcancelled\b']
    }
}

# Influence patterns
INFLUENCE_PATTERNS = {
    "control": [r'\b(?:controls?|controlling|controlled by)\b',
                r'\b(?:runs?|running|in charge of)\b',
                r'\b(?:owns?|owning|owned by)\b',
                r'\b(?:dominates?|dominating|dominated by)\b'],
    "manipulation": [r'\b(?:manipulates?|manipulating|manipulation)\b',
                     r'\b(?:orchestrates?|orchestrating|orchestrated)\b',
                     r'\b(?:pulling the strings|puppet master)\b',
                     r'\b(?:brainwash|indoctrinate|propaganda)\b'],
    "power": [r'\b(?:powerful|power over|wields? power)\b',
              r'\b(?:influence over|influential|sway)\b',
              r'\b(?:authority|authoritarian|dictator)\b'],
    "agency": [r'\b(?:behind|mastermind|architect of)\b',
               r'\b(?:responsible for|causing|created)\b',
               r'\b(?:engineered|designed|planned)\b'],
    "scope": [r'\b(?:everything|all of|entire|whole)\b',
              r'\b(?:global|worldwide|everywhere)\b',
              r'\b(?:complete control|total domination)\b']
}

# Key research focus entities
RESEARCH_FOCUS_ENTITIES = [
    'trump', 'biden', 'harris', 'putin', 'zelensky',
    'russia', 'ukraine', 'china', 'FBI', 'CIA',
    'deep state', 'globalists', 'epstein'
]

class CompletePoliticalDiscourseAnalyzer:
    def __init__(self):
        self.client = storage.Client(project=PROJECT_ID)
        self.bucket = self.client.bucket(BUCKET_NAME)

        # Initialize all tracking structures
        self._initialize_global_tracking()
        self._initialize_orientation_tracking()
        self._initialize_show_tracking()
        self._initialize_temporal_tracking()

    def _initialize_global_tracking(self):
        """Initialize global tracking structures"""
        self.entity_stats = defaultdict(lambda: {
            'blame': {'direct': 0, 'meta': 0, 'total': 0},
            'persuasive': defaultdict(lambda: defaultdict(int)),
            'influence': defaultdict(int),
            'contexts': []
        })

        self.category_stats = defaultdict(lambda: {
            'blame': {'direct': 0, 'meta': 0, 'total': 0},
            'persuasive': defaultdict(lambda: defaultdict(int)),
            'influence': defaultdict(int),
            'entity_breakdown': defaultdict(int)
        })

    def _initialize_orientation_tracking(self):
        """Initialize orientation-specific tracking"""
        self.orientation_entity_stats = {
            'Left wing': defaultdict(lambda: {
                'blame': {'direct': 0, 'meta': 0, 'total': 0},
                'persuasive': defaultdict(lambda: defaultdict(int)),
                'influence': defaultdict(int),
                'contexts': []
            }),
            'Right wing': defaultdict(lambda: {
                'blame': {'direct': 0, 'meta': 0, 'total': 0},
                'persuasive': defaultdict(lambda: defaultdict(int)),
                'influence': defaultdict(int),
                'contexts': []
            }),
            'Tenet': defaultdict(lambda: {
                'blame': {'direct': 0, 'meta': 0, 'total': 0},
                'persuasive': defaultdict(lambda: defaultdict(int)),
                'influence': defaultdict(int),
                'contexts': []
            })
        }

        self.orientation_category_stats = {
            'Left wing': defaultdict(lambda: {
                'blame': {'direct': 0, 'meta': 0, 'total': 0},
                'entity_breakdown': defaultdict(int)
            }),
            'Right wing': defaultdict(lambda: {
                'blame': {'direct': 0, 'meta': 0, 'total': 0},
                'entity_breakdown': defaultdict(int)
            }),
            'Tenet': defaultdict(lambda: {
                'blame': {'direct': 0, 'meta': 0, 'total': 0},
                'entity_breakdown': defaultdict(int)
            })
        }

        self.orientation_file_counts = {
            'Left wing': 0,
            'Right wing': 0,
            'Tenet': 0
        }

        self.orientation_stats = {
            'Left wing': defaultdict(lambda: {'blame': defaultdict(int)}),
            'Right wing': defaultdict(lambda: {'blame': defaultdict(int)}),
            'Tenet': defaultdict(lambda: {'blame': defaultdict(int)})
        }

    def _initialize_show_tracking(self):
        """Initialize show-level tracking"""
        self.show_stats = defaultdict(lambda: {
            'orientation': None,
            'episodes': 0,
            'total_blame': 0,
            'entity_blame': defaultdict(int),
            'category_blame': defaultdict(int),
            'temporal_data': defaultdict(lambda: {'episodes': 0, 'blame': 0}),
            'avg_blame_per_episode': 0,
            'influence_scores': [],
            # NEW: show-level persuasive counts
            'persuasive_counts': defaultdict(lambda: defaultdict(int))  # technique -> subtype -> int
        })

        self.orientation_show_breakdown = {
            'Left wing': defaultdict(int),
            'Right wing': defaultdict(int),
            'Tenet': defaultdict(int)
        }

    def _initialize_temporal_tracking(self):
        """Initialize temporal tracking"""
        self.temporal_stats = defaultdict(lambda: {
            'episodes': 0,
            'total_blame': 0,
            'influence_scores': [],
            'top_entities': defaultdict(int),
            'top_categories': defaultdict(int)
        })

        self.orientation_temporal_stats = {
            'Left wing': defaultdict(lambda: {'episodes': 0, 'total_blame': 0}),
            'Right wing': defaultdict(lambda: {'episodes': 0, 'total_blame': 0}),
            'Tenet': defaultdict(lambda: {'episodes': 0, 'total_blame': 0})
        }

    def extract_date_from_filename(self, file_path):
        """Extract date from filename - try multiple patterns"""
        # Try standard date pattern YYYY-MM-DD
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
        if date_match:
            return date_match.group(1)

        # Try YYYYMMDD pattern
        date_match = re.search(r'(\d{8})', file_path)
        if date_match:
            date_str = date_match.group(1)
            try:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            except Exception:
                pass

        # For Tenet files, default to a placeholder date if none found
        if 'Tenet_' in file_path:
            return "2024-01-01"  # Default date for Tenet files

        return None

    def extract_show_name(self, file_path):
        """Extract show name from file path - handle different formats"""
        # For standard format with double underscores
        parts = file_path.split('__')
        if len(parts) >= 3:
            return parts[1]

        # For Tenet files - extract the host name if present
        if 'Tenet_' in file_path:
            hosts = ['lauren_southern', 'matt_christiansen', 'tayler_hansen', 'tim_pool',
                     'benny_johnson', 'dave_rubin']
            for host in hosts:
                if host in file_path.lower():
                    return f"Tenet_{host}"
            return 'Tenet_show'

        return 'unknown_show'

    def find_political_keywords(self, text):
        """Find political entities with positions"""
        found = []

        # Check case-insensitive entities
        text_lower = text.lower()
        for category, entities in CASE_INSENSITIVE_ENTITIES.items():
            for entity in entities:
                pattern = rf'\b{re.escape(entity)}\b'
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in matches:
                    found.append({
                        'entity': entity,
                        'category': category,
                        'start': match.start(),
                        'end': match.end(),
                        'match_text': text[match.start():match.end()]
                    })

        # Check case-sensitive entities
        for category, entities in CASE_SENSITIVE_ENTITIES.items():
            for entity in entities:
                pattern = rf'\b{re.escape(entity)}\b'
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    found.append({
                        'entity': entity,
                        'category': category,
                        'start': match.start(),
                        'end': match.end(),
                        'match_text': match.group()
                    })

        # Check special entities
        for category, patterns in SPECIAL_ENTITIES.items():
            for entity_name, pattern in patterns:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    found.append({
                        'entity': entity_name,
                        'category': category,
                        'start': match.start(),
                        'end': match.end(),
                        'match_text': match.group()
                    })

        return found

    def analyze_blame(self, context, entity):
        """Analyze blame patterns in context"""
        context_lower = context.lower()
        blame_info = {
            'has_blame': False,
            'blame_type': None,
            'blame_phrases': []
        }

        # Direct blame indicators
        direct_patterns = [
            r'\b(?:is|are|was|were)\s+(?:directly\s+)?responsible\b',
            r'\b(?:is|are|was|were)\s+(?:to\s+)?blame\b',
            r'\bcaused\s+by\b',
            r'\bdue\s+to\b',
            r'\bbecause\s+of\b',
            r'\bfault of\b',
            r'\bthe reason for\b'
        ]

        # Meta-blame indicators
        meta_patterns = [
            r'\bthey\s+(?:always\s+)?blame\b',
            r'\bblame\s+(?:it\s+)?(?:all\s+)?on\b',
            r'\bscapegoat',
            r'\bblame\s+game\b',
            r'\bpointing\s+fingers?\b',
            r'\b(?:love|loves|quick|eager)\s+to\s+blame\b',
            r'\bresponsible for our problems\b',
            r'\bto blame for\b'
        ]

        for pattern in direct_patterns:
            if re.search(pattern, context_lower):
                blame_info['has_blame'] = True
                blame_info['blame_type'] = 'direct'
                blame_info['blame_phrases'].append(pattern)

        for pattern in meta_patterns:
            if re.search(pattern, context_lower):
                blame_info['has_blame'] = True
                blame_info['blame_type'] = 'meta' if not blame_info['blame_type'] else blame_info['blame_type']
                blame_info['blame_phrases'].append(pattern)

        return blame_info

    def analyze_persuasive_language(self, context):
        """Analyze persuasive language techniques"""
        persuasive_found = defaultdict(lambda: defaultdict(list))
        context_lower = context.lower()

        for technique, subtypes in PERSUASIVE_PATTERNS.items():
            for subtype, patterns in subtypes.items():
                for pattern in patterns:
                    matches = re.findall(pattern, context_lower, re.IGNORECASE)
                    if matches:
                        persuasive_found[technique][subtype].extend(matches)

        return persuasive_found

    def analyze_influence(self, context):
        """Analyze influence and power attribution"""
        influence_found = defaultdict(list)
        context_lower = context.lower()

        for influence_type, patterns in INFLUENCE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, context_lower, re.IGNORECASE)
                if matches:
                    influence_found[influence_type].extend(matches)

        return influence_found

    def calculate_influence_score(self, blame_count, persuasive_count, word_count):
        """Calculate influence intensity score"""
        if word_count == 0:
            return 0
        return round((blame_count + persuasive_count) / word_count * 1000, 2)

    def update_stats(self, instance, orientation, year_month, show_name):
        """Update all statistics including orientation and show-level"""
        entity = instance['entity']
        category = instance['category']

        # Update global stats
        if instance['blame']['has_blame']:
            blame_type = instance['blame']['blame_type']

            # Global entity stats
            self.entity_stats[entity]['blame'][blame_type] += 1
            self.entity_stats[entity]['blame']['total'] += 1

            # Global category stats
            self.category_stats[category]['blame'][blame_type] += 1
            self.category_stats[category]['blame']['total'] += 1
            self.category_stats[category]['entity_breakdown'][entity] += 1

            # Orientation-specific entity stats
            self.orientation_entity_stats[orientation][entity]['blame'][blame_type] += 1
            self.orientation_entity_stats[orientation][entity]['blame']['total'] += 1

            # Orientation-specific tracking (for normalization)
            self.orientation_stats[orientation][entity]['blame'][blame_type] += 1

            # Orientation category stats
            self.orientation_category_stats[orientation][category]['blame'][blame_type] += 1
            self.orientation_category_stats[orientation][category]['blame']['total'] += 1
            self.orientation_category_stats[orientation][category]['entity_breakdown'][entity] += 1

            # Show-level stats
            self.show_stats[show_name]['total_blame'] += 1
            self.show_stats[show_name]['entity_blame'][entity] += 1
            self.show_stats[show_name]['category_blame'][category] += 1
            self.show_stats[show_name]['temporal_data'][year_month]['blame'] += 1

            # Temporal stats
            self.temporal_stats[year_month]['total_blame'] += 1
            self.temporal_stats[year_month]['top_entities'][entity] += 1
            self.temporal_stats[year_month]['top_categories'][category] += 1
            self.orientation_temporal_stats[orientation][year_month]['total_blame'] += 1

        # Update persuasive language stats
        for technique, subtypes in instance['persuasive'].items():
            for subtype, matches in subtypes.items():
                if matches:
                    count = len(matches)
                    self.entity_stats[entity]['persuasive'][technique][subtype] += count
                    self.orientation_entity_stats[orientation][entity]['persuasive'][technique][subtype] += count
                    # NEW: show-level persuasive counts
                    self.show_stats[show_name]['persuasive_counts'][technique][subtype] += count

        # Update influence stats
        for influence_type, matches in instance['influence'].items():
            if matches:
                count = len(matches)
                self.entity_stats[entity]['influence'][influence_type] += count
                self.orientation_entity_stats[orientation][entity]['influence'][influence_type] += count

        # Store example contexts (limit to 5 per entity per orientation)
        if len(self.orientation_entity_stats[orientation][entity]['contexts']) < 5:
            self.orientation_entity_stats[orientation][entity]['contexts'].append({
                'context': instance['context'][:200] + '...',
                'blame': instance['blame']['blame_type'] if instance['blame']['has_blame'] else None,
                'date': instance['date'],
                'show': show_name
            })

    def process_file(self, file_path):
        """Process a single file with complete tracking"""
        try:
            blob = self.bucket.blob(file_path)
            content = blob.download_as_string()
            data = json.loads(content)

            segments = data.get('segments', [])
            if not segments:
                logger.warning(f"No segments found in {file_path}")
                return None

            # Extract metadata - improved for Tenet files
            if 'Tenet_' in file_path:
                orientation = 'Tenet'
            elif 'Right wing__' in file_path:
                orientation = 'Right wing'
            elif 'Left wing__' in file_path:
                orientation = 'Left wing'
            else:
                logger.warning(f"Unknown orientation for file: {file_path}")
                return None

            # Extract show name
            show_name = self.extract_show_name(file_path)

            # Extract date
            date = self.extract_date_from_filename(file_path)
            year_month = date[:7] if date else 'unknown'

            # Update file counts
            self.orientation_file_counts[orientation] += 1

            # Update show stats
            self.show_stats[show_name]['orientation'] = orientation
            self.show_stats[show_name]['episodes'] += 1
            self.orientation_show_breakdown[orientation][show_name] += 1

            # Update temporal stats
            self.temporal_stats[year_month]['episodes'] += 1
            self.orientation_temporal_stats[orientation][year_month]['episodes'] += 1
            self.show_stats[show_name]['temporal_data'][year_month]['episodes'] += 1

            file_results = {
                'orientation': orientation,
                'show_name': show_name,
                'date': date,
                'year_month': year_month,
                'instances': [],
                'word_count': sum(len(seg.get('text', '').split()) for seg in segments)
            }

            # Process each segment
            for seg_idx, segment in enumerate(segments):
                text = segment.get('text', '')
                if not text:
                    continue

                # Find political keywords
                keywords = self.find_political_keywords(text)

                for keyword_info in keywords:
                    # Extract context
                    kw_start = keyword_info['start']
                    kw_end = keyword_info['end']
                    context_start = max(0, kw_start - CONTEXT_WINDOW)
                    context_end = min(len(text), kw_end + CONTEXT_WINDOW)
                    context = text[context_start:context_end]

                    # Analyze blame
                    blame_info = self.analyze_blame(context, keyword_info['entity'])

                    # Analyze persuasive language
                    persuasive_info = self.analyze_persuasive_language(context)

                    # Analyze influence
                    influence_info = self.analyze_influence(context)

                    # Only record if something was found
                    if blame_info['has_blame'] or persuasive_info or influence_info:
                        instance = {
                            'segment': seg_idx,
                            'entity': keyword_info['entity'],
                            'category': keyword_info['category'],
                            'context': context,
                            'blame': blame_info,
                            'persuasive': dict(persuasive_info),
                            'influence': dict(influence_info),
                            'date': date,
                            'year_month': year_month
                        }
                        file_results['instances'].append(instance)

                        # Update all statistics
                        self.update_stats(instance, orientation, year_month, show_name)

            # Calculate file-level influence score
            if file_results['instances'] and file_results['word_count'] > 0:
                total_blame = sum(1 for inst in file_results['instances'] if inst['blame']['has_blame'])
                total_persuasive = sum(
                    sum(len(matches) for subtype_matches in inst['persuasive'].values()
                        for matches in subtype_matches.values())
                    for inst in file_results['instances']
                )
                file_results['influence_score'] = self.calculate_influence_score(
                    total_blame, total_persuasive, file_results['word_count']
                )

                # Store influence score for show
                self.show_stats[show_name]['influence_scores'].append(file_results['influence_score'])

                # Store for temporal analysis
                self.temporal_stats[year_month]['influence_scores'].append(file_results['influence_score'])
            else:
                file_results['influence_score'] = 0

            return file_results

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def get_normalized_stats(self, orientation):
        """Get normalized statistics per 100 episodes for fair comparison"""
        episodes = self.orientation_file_counts[orientation]
        if episodes == 0:
            return {}

        normalized = {}
        for entity, stats in self.orientation_entity_stats[orientation].items():
            if stats['blame']['total'] > 0:
                normalized[entity] = {
                    'blame_per_100': (stats['blame']['total'] / episodes) * 100,
                    'raw_blame': stats['blame']['total'],
                    'direct_per_100': (stats['blame']['direct'] / episodes) * 100,
                    'meta_per_100': (stats['blame']['meta'] / episodes) * 100
                }

        return normalized

    def run_batch(self, files, batch_name=""):
        """Process a batch of files"""
        results_by_orientation = defaultdict(int)

        for i, file_path in enumerate(files):
            if i % 10 == 0:
                logger.info(f"{batch_name}: Processing {i+1}/{len(files)}")

            file_results = self.process_file(file_path)

            if file_results and file_results['instances']:
                orientation = file_results['orientation']
                results_by_orientation[orientation] += 1

        return results_by_orientation

    def print_complete_summary(self):
        """Print comprehensive summary with all analytics"""
        print("\n" + "="*80)
        print("COMPLETE POLITICAL DISCOURSE ANALYSIS")
        print("="*80)

        # File counts
        print("\nFILES PROCESSED BY ORIENTATION:")
        print("-"*40)
        total_files = sum(self.orientation_file_counts.values())
        for orientation, count in self.orientation_file_counts.items():
            pct = (count / total_files * 100) if total_files > 0 else 0
            print(f"{orientation:12} : {count:4d} files ({pct:5.1f}%)")
        print(f"{'TOTAL':12} : {total_files:4d} files")

        # Global top blamed entities
        print("\n" + "="*80)
        print("GLOBAL TOP BLAMED ENTITIES (ALL ORIENTATIONS COMBINED)")
        print("="*80)
        blamed_entities = [(e, s['blame']['total']) for e, s in self.entity_stats.items()
                           if s['blame']['total'] > 0]
        blamed_entities.sort(key=lambda x: x[1], reverse=True)

        for entity, count in blamed_entities[:20]:
            stats = self.entity_stats[entity]
            print(f"{entity:20} - Total: {count:4d} (Direct: {stats['blame']['direct']:4d}, "
                  f"Meta: {stats['blame']['meta']:3d})")

        # Normalized comparison across orientations
        print("\n" + "="*80)
        print("NORMALIZED BLAME COMPARISON (PER 100 EPISODES)")
        print("="*80)
        print(f"{'Entity':<20} {'Left':<12} {'Right':<12} {'Tenet':<12}")
        print("-"*60)

        # Get normalized stats for each orientation
        left_norm = self.get_normalized_stats('Left wing')
        right_norm = self.get_normalized_stats('Right wing')
        tenet_norm = self.get_normalized_stats('Tenet')

        # Combine all entities
        all_entities = set()
        all_entities.update(left_norm.keys())
        all_entities.update(right_norm.keys())
        all_entities.update(tenet_norm.keys())

        # Sort by total normalized blame
        entity_totals = []
        for entity in all_entities:
            total = (left_norm.get(entity, {}).get('blame_per_100', 0) +
                     right_norm.get(entity, {}).get('blame_per_100', 0) +
                     tenet_norm.get(entity, {}).get('blame_per_100', 0))
            entity_totals.append((entity, total))

        entity_totals.sort(key=lambda x: x[1], reverse=True)

        for entity, _ in entity_totals[:20]:
            left_val = left_norm.get(entity, {}).get('blame_per_100', 0)
            right_val = right_norm.get(entity, {}).get('blame_per_100', 0)
            tenet_val = tenet_norm.get(entity, {}).get('blame_per_100', 0)
            print(f"{entity:<20} {left_val:<12.1f} {right_val:<12.1f} {tenet_val:<12.1f}")

        # Show-level analysis summary
        print("\n" + "="*80)
        print("TOP SHOWS BY BLAME INTENSITY")
        print("="*80)

        # Calculate average blame per episode for all shows
        show_intensity = []
        for show_name, stats in self.show_stats.items():
            if stats['episodes'] >= 1:  # Lower threshold for Tenet shows
                avg_blame = stats['total_blame'] / stats['episodes']
                show_intensity.append({
                    'name': show_name,
                    'orientation': stats['orientation'],
                    'episodes': stats['episodes'],
                    'avg_blame': avg_blame,
                    'total_blame': stats['total_blame']
                })

        show_intensity.sort(key=lambda x: x['avg_blame'], reverse=True)

        print(f"{'Show Name':<40} {'Orient':<8} {'Episodes':<10} {'Avg Blame/Ep':<12}")
        print("-"*80)
        for show in show_intensity[:15]:
            print(f"{show['name'][:40]:<40} {show['orientation']:<8} "
                  f"{show['episodes']:<10} {show['avg_blame']:<12.1f}")

        # Research focus entities
        print("\n" + "="*80)
        print("RESEARCH FOCUS ENTITIES - DETAILED BREAKDOWN")
        print("="*80)
        print(f"{'Entity':<15} {'Total':<8} {'Left':<8} {'Right':<8} {'Tenet':<8} {'Influence':<10}")
        print("-"*70)

        for entity in RESEARCH_FOCUS_ENTITIES:
            # handle case-insensitive lookup
            key_candidates = [entity, entity.lower(), entity.upper(), entity.title()]
            key = next((k for k in key_candidates if k in self.entity_stats), None)
            if not key:
                continue
            stats = self.entity_stats[key]
            if stats['blame']['total'] == 0:
                continue

            total_blame = stats['blame']['total']
            influence = sum(stats['influence'].values())

            # Orientation breakdown
            left_blame = right_blame = tenet_blame = 0
            for orientation, orient_stats in self.orientation_entity_stats.items():
                # try same key
                if key in orient_stats:
                    if orientation == 'Left wing':
                        left_blame = orient_stats[key]['blame']['total']
                    elif orientation == 'Right wing':
                        right_blame = orient_stats[key]['blame']['total']
                    elif orientation == 'Tenet':
                        tenet_blame = orient_stats[key]['blame']['total']
                else:
                    # try lower/upper variants
                    for k2 in [key.lower(), key.upper(), key.title()]:
                        if k2 in orient_stats:
                            if orientation == 'Left wing':
                                left_blame = orient_stats[k2]['blame']['total']
                            elif orientation == 'Right wing':
                                right_blame = orient_stats[k2]['blame']['total']
                            elif orientation == 'Tenet':
                                tenet_blame = orient_stats[k2]['blame']['total']
                            break

            print(f"{entity:<15} {total_blame:<8} {left_blame:<8} {right_blame:<8} "
                  f"{tenet_blame:<8} {influence:<10}")

    def save_complete_results(self):
        """Save all results with complete analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Orientation comparison CSV
        comparison_data = []
        all_entities = set()
        for orientation_stats in self.orientation_entity_stats.values():
            all_entities.update(orientation_stats.keys())

        for entity in all_entities:
            left_stats = self.orientation_entity_stats['Left wing'].get(entity, {'blame': {'total': 0}})
            right_stats = self.orientation_entity_stats['Right wing'].get(entity, {'blame': {'total': 0}})
            tenet_stats = self.orientation_entity_stats['Tenet'].get(entity, {'blame': {'total': 0}})

            total = left_stats['blame']['total'] + right_stats['blame']['total'] + tenet_stats['blame']['total']

            if total > 0:
                row = {
                    'entity': entity,
                    'total_blame': total,
                    'left_blame': left_stats['blame']['total'],
                    'right_blame': right_stats['blame']['total'],
                    'tenet_blame': tenet_stats['blame']['total'],
                    'left_normalized': (left_stats['blame']['total'] / self.orientation_file_counts['Left wing'] * 100)
                    if self.orientation_file_counts['Left wing'] > 0 else 0,
                    'right_normalized': (right_stats['blame']['total'] / self.orientation_file_counts['Right wing'] * 100)
                    if self.orientation_file_counts['Right wing'] > 0 else 0,
                    'tenet_normalized': (tenet_stats['blame']['total'] / self.orientation_file_counts['Tenet'] * 100)
                    if self.orientation_file_counts['Tenet'] > 0 else 0
                }
                comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.sort_values('total_blame', ascending=False, inplace=True)
            comparison_df.to_csv(f"{OUTPUT_DIR}/orientation_comparison_{timestamp}.csv", index=False)

        # 2. Show-level analysis CSV
        show_data = []
        show_intensity = []  # compute here for JSON
        for show_name, stats in self.show_stats.items():
            if stats['episodes'] > 0:
                top_entity = max(stats['entity_blame'].items(), key=lambda x: x[1])[0] if stats['entity_blame'] else 'none'
                top_category = max(stats['category_blame'].items(), key=lambda x: x[1])[0] if stats['category_blame'] else 'none'
                avg_influence = sum(stats['influence_scores']) / len(stats['influence_scores']) if stats['influence_scores'] else 0
                avg_blame = stats['total_blame'] / stats['episodes']

                row = {
                    'show_name': show_name,
                    'orientation': stats['orientation'],
                    'episodes': stats['episodes'],
                    'total_blame': stats['total_blame'],
                    'avg_blame_per_episode': avg_blame,
                    'avg_influence_score': avg_influence,
                    'top_blamed_entity': top_entity,
                    'top_category': top_category
                }
                show_data.append(row)
                show_intensity.append({
                    'name': show_name,
                    'orientation': stats['orientation'],
                    'episodes': stats['episodes'],
                    'avg_blame': avg_blame,
                    'total_blame': stats['total_blame']
                })

        if show_data:
            show_df = pd.DataFrame(show_data)
            show_df.sort_values(['orientation', 'avg_blame_per_episode'], ascending=[True, False], inplace=True)
            show_df.to_csv(f"{OUTPUT_DIR}/show_analysis_{timestamp}.csv", index=False)

        show_intensity.sort(key=lambda x: x['avg_blame'], reverse=True)

        # 3. Temporal analysis CSV
        temporal_data = []
        for month, data in self.temporal_stats.items():
            if month != 'unknown' and data['episodes'] > 0:
                avg_influence = sum(data['influence_scores']) / len(data['influence_scores']) if data['influence_scores'] else 0
                top_entity = max(data['top_entities'].items(), key=lambda x: x[1])[0] if data['top_entities'] else 'none'

                row = {
                    'year_month': month,
                    'total_episodes': data['episodes'],
                    'total_blame': data['total_blame'],
                    'avg_blame_per_episode': data['total_blame'] / data['episodes'],
                    'avg_influence_score': avg_influence,
                    'top_entity': top_entity
                }

                # Add orientation breakdown
                for orientation in ['Left wing', 'Right wing', 'Tenet']:
                    orient_data = self.orientation_temporal_stats[orientation].get(month, {})
                    row[f'{orientation.lower().replace(" ", "_")}_episodes'] = orient_data.get('episodes', 0)
                    row[f'{orientation.lower().replace(" ", "_")}_blame'] = orient_data.get('total_blame', 0)

                temporal_data.append(row)

        if temporal_data:
            temporal_df = pd.DataFrame(temporal_data)
            temporal_df.sort_values('year_month', inplace=True)
            temporal_df.to_csv(f"{OUTPUT_DIR}/temporal_analysis_{timestamp}.csv", index=False)

        # 4. Comprehensive JSON summary
        def _safe_stats(entity_key):
            # Return (blame_total, influence_total) case-tolerantly
            for k in (entity_key, entity_key.lower(), entity_key.upper(), entity_key.title()):
                if k in self.entity_stats:
                    return (self.entity_stats[k]['blame']['total'],
                            sum(self.entity_stats[k]['influence'].values()))
            return (0, 0)

        summary = {
            'timestamp': timestamp,
            'file_counts': self.orientation_file_counts,
            'total_files': sum(self.orientation_file_counts.values()),
            'normalized_comparison': {
                orientation: self.get_normalized_stats(orientation)
                for orientation in ['Left wing', 'Right wing', 'Tenet']
            },
            'top_shows_by_intensity': show_intensity[:20],
            'research_entities': {
                entity: {
                    'total_blame': _safe_stats(entity)[0],
                    'influence_total': _safe_stats(entity)[1],
                    'orientation_breakdown': {
                        orientation: (
                            self.orientation_entity_stats[orientation].get(entity, {'blame': {'total': 0}})['blame']['total']
                            if entity in self.orientation_entity_stats[orientation]
                            else self.orientation_entity_stats[orientation].get(entity.lower(), {'blame': {'total': 0}})['blame']['total']
                            if entity.lower() in self.orientation_entity_stats[orientation]
                            else self.orientation_entity_stats[orientation].get(entity.upper(), {'blame': {'total': 0}})['blame']['total']
                            if entity.upper() in self.orientation_entity_stats[orientation]
                            else self.orientation_entity_stats[orientation].get(entity.title(), {'blame': {'total': 0}})['blame']['total']
                        )
                        for orientation in ['Left wing', 'Right wing', 'Tenet']
                    }
                }
                for entity in RESEARCH_FOCUS_ENTITIES
            }
        }

        with open(f"{OUTPUT_DIR}/complete_analysis_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Complete results saved to {OUTPUT_DIR}/")
        return timestamp

    # NEW ENHANCED REPORTING METHODS
    def generate_enhanced_reports(self):
        """Generate comprehensive reports matching timeline script detail"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Detailed Entity Report with Persuasive Techniques
        self.generate_entity_persuasion_report(timestamp)

        # 2. Blame Attribution Patterns Report
        self.generate_blame_patterns_report(timestamp)

        # 3. Show-Level Influence Fingerprints
        self.generate_show_fingerprints_report(timestamp)

        # 4. Temporal Evolution Analysis
        self.generate_temporal_evolution_report(timestamp)

        # 5. Cross-Orientation Comparative Analysis
        self.generate_comparative_analysis_report(timestamp)

        # 6. Research Entity Deep Dive
        self.generate_research_entity_report(timestamp)

        logger.info(f"Enhanced reports generated with timestamp {timestamp}")

    def generate_entity_persuasion_report(self, timestamp):
        """Detailed entity analysis with persuasive technique breakdowns"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_lines = []
        report_lines.append("COMPREHENSIVE ENTITY ANALYSIS WITH PERSUASIVE TECHNIQUES")
        report_lines.append("="*80)

        top_entities = sorted(self.entity_stats.items(),
                              key=lambda x: x[1]['blame']['total'],
                              reverse=True)[:50]

        for entity, stats in top_entities:
            report_lines.append(f"\n{entity.upper()}")
            report_lines.append("-"*40)

            # Blame breakdown
            blame = stats['blame']
            report_lines.append("Blame Attribution:")
            report_lines.append(f"  Total: {blame['total']:4d} | Direct: {blame['direct']:4d} | Meta: {blame['meta']:4d}")

            # Blame by orientation
            report_lines.append("  By Orientation:")
            for orientation in ['Left wing', 'Right wing', 'Tenet']:
                orient_blame = self.orientation_entity_stats[orientation].get(entity, {}).get('blame', {}).get('total', 0)
                if orient_blame > 0:
                    report_lines.append(f"    {orientation}: {orient_blame:4d}")

            # Persuasive techniques used with this entity
            report_lines.append("\nPersuasive Techniques:")
            for technique, subtypes in stats['persuasive'].items():
                total_technique = sum(subtypes.values()) if isinstance(subtypes, dict) else 0
                total_technique = 0
                for _, counts in subtypes.items():
                    total_technique += counts if isinstance(counts, int) else sum(counts.values())
                if total_technique > 0:
                    report_lines.append(f"  {technique}: {total_technique}")
                    for subtype, counts in subtypes.items():
                        count_val = counts if isinstance(counts, int) else sum(counts.values())
                        if count_val > 0:
                            report_lines.append(f"    - {subtype}: {count_val}")

            # Influence patterns
            if stats['influence']:
                report_lines.append("\nInfluence Attribution:")
                for influence_type, count in stats['influence'].items():
                    if count > 0:
                        report_lines.append(f"  {influence_type}: {count}")

            # Sample contexts
            if stats.get('contexts'):
                report_lines.append("\nSample Contexts:")
                for i, ctx in enumerate(stats['contexts'][:3]):
                    report_lines.append(f"  {i+1}. \"{ctx['context'][:150]}...\"")
                    report_lines.append(f"     - Show: {ctx.get('show', 'unknown')}, Date: {ctx.get('date', 'unknown')}")

        with open(f"{OUTPUT_DIR}/entity_persuasion_analysis_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def generate_blame_patterns_report(self, timestamp):
        """Analyze blame attribution patterns across orientations"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_lines = []
        report_lines.append("BLAME ATTRIBUTION PATTERNS ANALYSIS")
        report_lines.append("="*80)

        # Category-level blame patterns
        report_lines.append("\nBLAME BY CATEGORY AND ORIENTATION")
        report_lines.append("-"*80)

        for category in sorted(self.category_stats.keys()):
            report_lines.append(f"\n{category.upper()}")
            report_lines.append(f"Global Total: {self.category_stats[category]['blame']['total']}")

            # Orientation breakdown
            for orientation in ['Left wing', 'Right wing', 'Tenet']:
                orient_stats = self.orientation_category_stats[orientation].get(category, {})
                if orient_stats.get('blame', {}).get('total', 0) > 0:
                    total = orient_stats['blame']['total']
                    direct = orient_stats['blame']['direct']
                    meta = orient_stats['blame']['meta']

                    report_lines.append(f"\n  {orientation}:")
                    report_lines.append(f"    Total: {total} (Direct: {direct}, Meta: {meta})")

                    # Top blamed entities in this category
                    entity_breakdown = orient_stats.get('entity_breakdown', {})
                    if entity_breakdown:
                        report_lines.append("    Top entities:")
                        for entity, count in sorted(entity_breakdown.items(),
                                                    key=lambda x: x[1],
                                                    reverse=True)[:5]:
                            report_lines.append(f"      - {entity}: {count}")

        # Blame type analysis
        report_lines.append("\n\nBLAME TYPE ANALYSIS")
        report_lines.append("-"*80)

        for orientation in ['Left wing', 'Right wing', 'Tenet']:
            total_direct = sum(e['blame']['direct'] for e in self.orientation_entity_stats[orientation].values())
            total_meta = sum(e['blame']['meta'] for e in self.orientation_entity_stats[orientation].values())
            total_blame = total_direct + total_meta

            if total_blame > 0:
                direct_pct = (total_direct / total_blame) * 100
                meta_pct = (total_meta / total_blame) * 100

                report_lines.append(f"\n{orientation}:")
                report_lines.append(f"  Direct blame: {total_direct:4d} ({direct_pct:5.1f}%)")
                report_lines.append(f"  Meta blame:   {total_meta:4d} ({meta_pct:5.1f}%)")

        with open(f"{OUTPUT_DIR}/blame_patterns_analysis_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def generate_show_fingerprints_report(self, timestamp):
        """Create detailed show-level fingerprints with influence calculations"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_lines = []
        report_lines.append("SHOW-LEVEL RHETORICAL FINGERPRINTS")
        report_lines.append("="*80)

        # Calculate comprehensive show metrics
        show_fingerprints = []

        for show_name, stats in self.show_stats.items():
            if stats['episodes'] == 0:
                continue

            # Influence score components
            avg_blame_per_ep = stats['total_blame'] / stats['episodes']
            avg_influence_score = sum(stats['influence_scores']) / len(stats['influence_scores']) if stats['influence_scores'] else 0

            # Dominant entities/categories
            top_entity = max(stats['entity_blame'].items(), key=lambda x: x[1])[0] if stats['entity_blame'] else 'none'
            top_category = max(stats['category_blame'].items(), key=lambda x: x[1])[0] if stats['category_blame'] else 'none'

            # Persuasive technique distribution (now show-level, not orientation-wide)
            persuasive_counts = defaultdict(int)
            for technique, by_subtype in stats['persuasive_counts'].items():
                for subtype, count_val in by_subtype.items():
                    if count_val > 0:
                        persuasive_counts[f"{technique}:{subtype}"] += count_val

            top_persuasive = sorted(persuasive_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            fingerprint = {
                'show': show_name,
                'orientation': stats['orientation'],
                'episodes': stats['episodes'],
                'avg_blame_per_ep': avg_blame_per_ep,
                'avg_influence_score': avg_influence_score,
                'top_entity': top_entity,
                'top_category': top_category,
                'top_persuasive': top_persuasive,
                'temporal_variance': self.calculate_temporal_variance(stats['temporal_data'])
            }
            show_fingerprints.append(fingerprint)

        # Sort by influence metrics
        show_fingerprints.sort(key=lambda x: x['avg_blame_per_ep'], reverse=True)

        # Generate detailed report
        for rank, fp in enumerate(show_fingerprints, 1):
            report_lines.append(f"\nRank {rank}: {fp['show']}")
            report_lines.append("-"*60)
            report_lines.append(f"  Orientation: {fp['orientation']}")
            report_lines.append(f"  Episodes: {fp['episodes']}")
            report_lines.append(f"  Avg Blame/Episode: {fp['avg_blame_per_ep']:.2f}")
            report_lines.append(f"  Avg Influence Score: {fp['avg_influence_score']:.2f}")
            report_lines.append(f"  Top Blamed Entity: {fp['top_entity']}")
            report_lines.append(f"  Top Category: {fp['top_category']}")
            report_lines.append("  Dominant Persuasive Techniques:")
            for tech, count in fp['top_persuasive']:
                report_lines.append(f"    - {tech}: {count}")
            report_lines.append(f"  Temporal Consistency (σ): {fp['temporal_variance']:.2f}")

        with open(f"{OUTPUT_DIR}/show_fingerprints_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def generate_temporal_evolution_report(self, timestamp):
        """Detailed temporal analysis with monthly breakdowns"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_lines = []
        report_lines.append("TEMPORAL EVOLUTION ANALYSIS")
        report_lines.append("="*80)

        # Sort months chronologically
        sorted_months = sorted([m for m in self.temporal_stats.keys() if m != 'unknown'])

        if not sorted_months:
            report_lines.append("No temporal data available")
            with open(f"{OUTPUT_DIR}/temporal_evolution_{timestamp}.txt", 'w') as f:
                f.write('\n'.join(report_lines))
            return

        # Monthly detailed breakdown
        report_lines.append("\nMONTHLY BLAME INTENSITY AND INFLUENCE METRICS")
        report_lines.append("-"*80)

        for month in sorted_months:
            data = self.temporal_stats[month]
            if data['episodes'] == 0:
                continue

            report_lines.append(f"\n{month}")
            report_lines.append(f"  Episodes: {data['episodes']}")
            report_lines.append(f"  Total Blame Instances: {data['total_blame']}")
            report_lines.append(f"  Blame per Episode: {data['total_blame'] / data['episodes']:.2f}")

            if data['influence_scores']:
                avg_influence = sum(data['influence_scores']) / len(data['influence_scores'])
                report_lines.append(f"  Avg Influence Score: {avg_influence:.2f}")

                # Compare to Tenet baseline
                tenet_similarity = 100 - abs((avg_influence - 5.45) / 5.45 * 100)
                report_lines.append(f"  Similarity to Tenet: {tenet_similarity:.1f}%")

            # Top entities this month
            if data['top_entities']:
                report_lines.append("  Top Blamed Entities:")
                for entity, count in sorted(data['top_entities'].items(),
                                            key=lambda x: x[1],
                                            reverse=True)[:5]:
                    report_lines.append(f"    - {entity}: {count}")

            # Orientation breakdown
            report_lines.append("  By Orientation:")
            for orientation in ['Left wing', 'Right wing', 'Tenet']:
                orient_data = self.orientation_temporal_stats[orientation].get(month, {})
                if orient_data.get('episodes', 0) > 0:
                    report_lines.append(f"    {orientation}: {orient_data['episodes']} episodes, "
                                        f"{orient_data['total_blame']} blame instances")

        # Trend analysis
        report_lines.append("\n\nTREND ANALYSIS")
        report_lines.append("-"*80)

        if len(sorted_months) >= 3:
            third = max(1, len(sorted_months) // 3)
            early_months = sorted_months[:third]
            late_months = sorted_months[-third:]

            def _sum_blame_ep(months):
                b_sum, ep_sum = 0, 0
                for m in months:
                    if self.temporal_stats[m]['episodes'] > 0:
                        b_sum += self.temporal_stats[m]['total_blame']
                        ep_sum += self.temporal_stats[m]['episodes']
                return b_sum, ep_sum

            early_blame_sum, early_episodes = _sum_blame_ep(early_months)
            late_blame_sum, late_episodes = _sum_blame_ep(late_months)

            if early_episodes > 0 and late_episodes > 0:
                early_blame_avg = early_blame_sum / early_episodes
                late_blame_avg = late_blame_sum / late_episodes
                trend_pct = ((late_blame_avg - early_blame_avg) / early_blame_avg) * 100 if early_blame_avg > 0 else 0

                report_lines.append(f"Early period average: {early_blame_avg:.2f} blame/episode")
                report_lines.append(f"Late period average: {late_blame_avg:.2f} blame/episode")
                report_lines.append(f"Trend: {'+' if trend_pct > 0 else ''}{trend_pct:.1f}%")

        with open(f"{OUTPUT_DIR}/temporal_evolution_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def generate_research_entity_report(self, timestamp):
        """Deep dive into research focus entities"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_lines = []
        report_lines.append("RESEARCH FOCUS ENTITIES - COMPREHENSIVE ANALYSIS")
        report_lines.append("="*80)

        for entity in RESEARCH_FOCUS_ENTITIES:
            # Try multiple case variants
            key_candidates = [entity, entity.lower(), entity.upper(), entity.title()]
            key = next((k for k in key_candidates if k in self.entity_stats), None)
            if not key:
                continue

            stats = self.entity_stats[key]
            if stats['blame']['total'] == 0:
                continue

            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"{entity.upper()}")
            report_lines.append(f"{'='*80}")

            # Overall statistics
            report_lines.append("\nOVERALL STATISTICS:")
            report_lines.append(f"  Total Blame: {stats['blame']['total']}")
            report_lines.append(f"  Direct: {stats['blame']['direct']} | Meta: {stats['blame']['meta']}")
            report_lines.append(f"  Influence Attributions: {sum(stats['influence'].values())}")

            # Orientation comparison
            report_lines.append("\nBY ORIENTATION:")
            for orientation in ['Left wing', 'Right wing', 'Tenet']:
                orient_stats = self.orientation_entity_stats[orientation].get(key, {})
                if not orient_stats:
                    # try variants
                    for k2 in [key.lower(), key.upper(), key.title()]:
                        orient_stats = self.orientation_entity_stats[orientation].get(k2, {})
                        if orient_stats:
                            break

                if orient_stats.get('blame', {}).get('total', 0) > 0:
                    blame = orient_stats['blame']
                    episodes = self.orientation_file_counts[orientation]
                    normalized = (blame['total'] / episodes * 100) if episodes > 0 else 0

                    report_lines.append(f"\n  {orientation}:")
                    report_lines.append(f"    Raw blame: {blame['total']} (D:{blame['direct']}, M:{blame['meta']})")
                    report_lines.append(f"    Normalized per 100 episodes: {normalized:.1f}")

                    # Persuasive techniques
                    if orient_stats.get('persuasive'):
                        report_lines.append("    Persuasive techniques used:")
                        for technique, subtypes in orient_stats['persuasive'].items():
                            total = 0
                            for _, counts in subtypes.items():
                                total += counts if isinstance(counts, int) else sum(counts.values())
                            if total > 0:
                                report_lines.append(f"      - {technique}: {total}")

                    # Sample contexts
                    if orient_stats.get('contexts'):
                        report_lines.append("    Sample contexts:")
                        for ctx in orient_stats['contexts'][:2]:
                            report_lines.append(f"      \"{ctx['context'][:100]}...\"")
                            report_lines.append(f"        ({ctx['show']}, {ctx['date']})")

            # Co-occurrence analysis (placeholder; requires segment-level cross-entity tracking)
            report_lines.append("\nCO-OCCURRENCE ANALYSIS:")
            co_occurrences = self.analyze_entity_co_occurrences(entity)
            if co_occurrences:
                for co_entity, count in sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report_lines.append(f"  Often blamed with {co_entity}: {count} times")
            else:
                report_lines.append("  No co-occurrence data available")

        with open(f"{OUTPUT_DIR}/research_entities_analysis_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def generate_comparative_analysis_report(self, timestamp):
        """Cross-orientation comparative analysis with statistical insights"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_lines = []
        report_lines.append("CROSS-ORIENTATION COMPARATIVE ANALYSIS")
        report_lines.append("="*80)

        # Calculate orientation-level statistics
        orientation_stats = {}
        for orientation in ['Left wing', 'Right wing', 'Tenet']:
            episodes = self.orientation_file_counts[orientation]
            if episodes == 0:
                continue

            # Aggregate metrics
            total_blame = sum(e['blame']['total'] for e in self.orientation_entity_stats[orientation].values())
            unique_entities = len([e for e in self.orientation_entity_stats[orientation]
                                   if self.orientation_entity_stats[orientation][e]['blame']['total'] > 0])

            # Persuasive technique distribution
            persuasive_dist = defaultdict(int)
            for entity_stats in self.orientation_entity_stats[orientation].values():
                for technique, subtypes in entity_stats['persuasive'].items():
                    for subtype, counts in subtypes.items():
                        count_val = counts if isinstance(counts, int) else sum(counts.values())
                        persuasive_dist[technique] += count_val

            orientation_stats[orientation] = {
                'episodes': episodes,
                'total_blame': total_blame,
                'blame_per_episode': total_blame / episodes if episodes > 0 else 0,
                'unique_entities': unique_entities,
                'entities_per_episode': unique_entities / episodes if episodes > 0 else 0,
                'persuasive_distribution': dict(persuasive_dist)
            }

        # Generate comparative report
        report_lines.append("\nORIENTATION METRICS COMPARISON")
        report_lines.append("-"*80)
        report_lines.append(f"{'Metric':<30} {'Left wing':<15} {'Right wing':<15} {'Tenet':<15}")
        report_lines.append("-"*80)

        metrics = [
            ('Episodes', 'episodes'),
            ('Total Blame', 'total_blame'),
            ('Blame per Episode', 'blame_per_episode'),
            ('Unique Entities Blamed', 'unique_entities'),
            ('Entities per Episode', 'entities_per_episode')
        ]

        for label, key in metrics:
            row = f"{label:<30}"
            for orientation in ['Left wing', 'Right wing', 'Tenet']:
                if orientation in orientation_stats:
                    value = orientation_stats[orientation].get(key, 0)
                    if isinstance(value, float):
                        row += f"{value:<15.2f}"
                    else:
                        row += f"{value:<15d}"
                else:
                    row += f"{'N/A':<15}"
            report_lines.append(row)

        # Persuasive technique comparison
        report_lines.append("\n\nPERSUASIVE TECHNIQUE DISTRIBUTION")
        report_lines.append("-"*80)

        all_techniques = set()
        for stats in orientation_stats.values():
            all_techniques.update(stats['persuasive_distribution'].keys())

        for technique in sorted(all_techniques):
            report_lines.append(f"\n{technique}:")
            for orientation in ['Left wing', 'Right wing', 'Tenet']:
                if orientation in orientation_stats:
                    count = orientation_stats[orientation]['persuasive_distribution'].get(technique, 0)
                    episodes = orientation_stats[orientation]['episodes']
                    per_100 = (count / episodes * 100) if episodes > 0 else 0
                    report_lines.append(f"  {orientation}: {count} total ({per_100:.1f} per 100 episodes)")

        # Entity overlap analysis
        report_lines.append("\n\nENTITY BLAME OVERLAP ANALYSIS")
        report_lines.append("-"*80)

        # Find entities blamed by multiple orientations
        entity_orientations = defaultdict(set)
        for orientation in ['Left wing', 'Right wing', 'Tenet']:
            for entity, stats in self.orientation_entity_stats[orientation].items():
                if stats['blame']['total'] > 0:
                    entity_orientations[entity].add(orientation)

        # Categorize by overlap
        all_three = [e for e, orients in entity_orientations.items() if len(orients) == 3]
        two_orient = [e for e, orients in entity_orientations.items() if len(orients) == 2]
        one_orient = [e for e, orients in entity_orientations.items() if len(orients) == 1]

        report_lines.append(f"Entities blamed by all three orientations: {len(all_three)}")
        for entity in sorted(all_three)[:10]:
            report_lines.append(f"  - {entity}")

        report_lines.append(f"\nEntities blamed by two orientations: {len(two_orient)}")

        report_lines.append(f"\nOrientation-exclusive entities:")
        for orientation in ['Left wing', 'Right wing', 'Tenet']:
            exclusive = [e for e in one_orient if list(entity_orientations[e])[0] == orientation]
            report_lines.append(f"  {orientation} only: {len(exclusive)}")
            for entity in sorted(exclusive)[:5]:
                report_lines.append(f"    - {entity}")

        with open(f"{OUTPUT_DIR}/comparative_analysis_{timestamp}.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    # Helper methods
    def calculate_temporal_variance(self, temporal_data):
        """Calculate temporal consistency metric (std dev of blame/episode across months)"""
        if not temporal_data or len(temporal_data) < 2:
            return 0.0

        monthly_vals = []
        for d in temporal_data.values():
            episodes = d.get('episodes', 0)
            monthly_vals.append((d.get('blame', 0) / episodes) if episodes > 0 else 0)

        if not monthly_vals:
            return 0.0

        mean = sum(monthly_vals) / len(monthly_vals)
        variance = sum((x - mean) ** 2 for x in monthly_vals) / len(monthly_vals)
        return variance ** 0.5  # Standard deviation

    def analyze_entity_co_occurrences(self, target_entity):
        """Find entities that co-occur with target entity (placeholder: requires segment-level mapping)"""
        # Not tracked in this script; return empty for now.
        return defaultdict(int)


def main():
    analyzer = CompletePoliticalDiscourseAnalyzer()

    # Get all files
    all_blobs = list(analyzer.bucket.list_blobs(prefix='enriched_transcripts/'))
    json_files = [b.name for b in all_blobs if b.name.endswith('_enriched.json')]

    # Filter for different orientations
    tenet_files = [f for f in json_files if 'Tenet_' in f]
    left_files = [f for f in json_files if 'Left wing__' in f]
    right_files = [f for f in json_files if 'Right wing__' in f]

    print(f"Files found:")
    print(f"  Tenet: {len(tenet_files)}")
    print(f"  Left wing: {len(left_files)}")
    print(f"  Right wing: {len(right_files)}")
    print(f"  Total: {len(json_files)}")

    # Test with a small batch first
    print("\n" + "="*60)
    print("Testing with small batch from each orientation...")

    test_files = []
    # Get 5 files from each orientation for testing
    test_files.extend(tenet_files[:5] if tenet_files else [])
    test_files.extend(left_files[:5] if left_files else [])
    test_files.extend(right_files[:5] if right_files else [])

    print(f"\nProcessing test batch of {len(test_files)} files...")

    # Process test batch
    results = analyzer.run_batch(test_files, "Test batch")

    # Print summary
    analyzer.print_complete_summary()

    # Ask whether to continue
    print("\n" + "="*60)
    print(f"Test batch processed. Found instances in:")
    for ori, count in results.items():
        print(f"  {ori}: {count} files")

    print("\n" + "="*60)
    response = input("Continue with all files? (y/n): ")

    if response.lower() == 'y':
        # Process all files
        print(f"\nProcessing all {len(json_files)} files...")

        # Reset analyzer for full run
        analyzer = CompletePoliticalDiscourseAnalyzer()

        # Process in chunks
        chunk_size = 500
        for i in range(0, len(json_files), chunk_size):
            chunk = json_files[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(json_files) + chunk_size - 1) // chunk_size

            print(f"\nProcessing chunk {chunk_num}/{total_chunks}...")
            analyzer.run_batch(chunk, f"Chunk {chunk_num}")

        # Print final summary
        analyzer.print_complete_summary()

        # Save results
        timestamp = analyzer.save_complete_results()

        # Generate enhanced reports
        print("\nGenerating enhanced analysis reports...")
        analyzer.generate_enhanced_reports()

        # Upload to GCS
        print("\nUploading results to GCS...")
        os.system(f'gsutil -m cp -r {OUTPUT_DIR}/*{timestamp}* gs://{BUCKET_NAME}/{OUTPUT_DIR}/')
        print("Upload complete!")
    else:
        print("Stopping after test batch.")
        # Save test results
        timestamp = analyzer.save_complete_results()

        # Generate enhanced reports for test batch
        print("\nGenerating enhanced analysis reports for test batch...")
        analyzer.generate_enhanced_reports()

        print(f"Test results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()