#!/usr/bin/env python3
"""
TRANSPARENT BOOTSTRAP ANALYSIS FROM ENHANCED DATA FILES
=======================================================

This script performs bootstrap confidence interval analysis using the enhanced
analysis files that contain pre-calculated influence scores.

DATA PIPELINE DOCUMENTATION:
1. Raw podcasts → Whisper transcription → enriched_transcripts/
2. Enriched transcripts → enhanced analysis scripts → enhanced_analysis/ files
3. Enhanced analysis files → THIS SCRIPT → bootstrap confidence intervals

CALCULATION METHOD:
The influence scores in the enhanced analysis files were calculated as:
influence_ratio = (causal_count + persuasive_count) / word_count * 1000

Where:
- causal_count = number of causal patterns found via regex matching
- persuasive_count = number of persuasive technique keywords found
- word_count = total words in episode
- Multiplied by 1000 to get "per 1000 words" metric

Author: [Your name]
Date: August 2025
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import os
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from storage_backend import destination, is_local, storage  # local folder or GCS; see storage_backend.py
import pandas as pd

# Configuration
PROJECT_ID = "handy-vortex-459018-g0"
BUCKET_NAME = "podcast-dissertation-audio"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42  # For reproducibility

# Set random seed
np.random.seed(RANDOM_SEED)

def document_data_source():
    """Document exactly where the data comes from."""
    documentation = """
    DATA SOURCE DOCUMENTATION
    ========================
    
    1. Enhanced Analysis Files Location:
       - Tenet: gs://podcast-dissertation-audio/tenet_enhanced_analysis/
       - US Podcasts: gs://podcast-dissertation-audio/us_podcast_enhanced_analysis/
    
    2. File Structure:
       - Each file contains a JSON list of episode objects
       - Typically 25 episodes per batch file
       - Total expected: ~405 Tenet + ~6,027 US episodes
    
    3. Influence Score Calculation (from enhanced analysis):
       influence_ratio = (causal_count + persuasive_count) / word_count * 1000
       
       This represents the density of influence techniques per 1000 words.
    
    4. Data Integrity:
       - Each episode has a unique 'source_file' identifier
       - Duplicates are removed using this identifier
       - Bootstrap sampling maintains episode-level independence
    """
    print(documentation)
    return documentation

def extract_show_and_orientation(filename):
    """Extract show name and political orientation from filename.
    
    Examples:
    - tenet_enhanced_batch_001_20250802_103747.json → ('tenet_media', 'Tenet')
    - Right_wing_thejoeroganexperience_enhanced_batch_001.json → ('thejoeroganexperience', 'Right wing')
    """
    filename = filename.split('/')[-1]
    
    if 'tenet' in filename.lower():
        return 'tenet_media', 'Tenet'
    
    parts = filename.split('_')
    try:
        # Find where 'enhanced' appears
        enhanced_idx = None
        for i, part in enumerate(parts):
            if part == 'enhanced':
                enhanced_idx = i
                break
        
        if enhanced_idx and enhanced_idx >= 3:
            orientation = f"{parts[0]} {parts[1]}"  # "Right wing" or "Left wing"
            show_name = '_'.join(parts[2:enhanced_idx])
            return show_name, orientation
    except Exception as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
    
    return 'unknown', 'unknown'

def load_all_episodes_with_validation():
    """Load all episodes from enhanced analysis files with detailed validation."""
    print("\n📊 LOADING DATA WITH FULL TRANSPARENCY")
    print("=" * 50)
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # Track everything for transparency
    all_episodes = []
    seen_source_files = set()
    duplicate_count = 0
    file_counts = {'tenet': 0, 'us': 0}
    episode_counts = defaultdict(int)
    
    # Process Tenet files
    print("\n1. Processing Tenet Media files...")
    tenet_blobs = list(bucket.list_blobs(prefix='tenet_enhanced_analysis/'))
    tenet_batch_files = [b for b in tenet_blobs if b.name.endswith('.json') and 'batch' in b.name]
    
    print(f"   Found {len(tenet_batch_files)} Tenet batch files")
    
    for blob in tenet_batch_files:
        try:
            content = blob.download_as_string()
            episodes = json.loads(content)
            file_counts['tenet'] += 1
            
            for episode in episodes:
                source_file = episode.get('metadata', {}).get('source_file', '')
                
                # Check for duplicates
                if source_file in seen_source_files:
                    duplicate_count += 1
                    continue
                
                seen_source_files.add(source_file)
                
                # Extract influence score
                influence_score = episode.get('influence_metrics', {}).get('influence_ratio')
                if influence_score is None:
                    print(f"   Warning: No influence score in {source_file}")
                    continue
                
                # Add episode with metadata
                all_episodes.append({
                    'influence_score': float(influence_score),
                    'orientation': 'Tenet',
                    'show_name': 'tenet_media',
                    'source_file': source_file,
                    'word_count': episode['metadata']['total_word_count'],
                    'causal_density': episode['influence_metrics']['causal_density_per_1k'],
                    'persuasive_density': episode['influence_metrics']['persuasive_density_per_1k']
                })
                episode_counts['Tenet'] += 1
                
        except Exception as e:
            print(f"   Error processing {blob.name}: {e}")
    
    print(f"   Loaded {episode_counts['Tenet']} unique Tenet episodes")
    
    # Process US podcast files
    print("\n2. Processing US podcast files...")
    us_blobs = list(bucket.list_blobs(prefix='us_podcast_enhanced_analysis/'))
    us_batch_files = [b for b in us_blobs if b.name.endswith('.json') and 'batch' in b.name]
    
    print(f"   Found {len(us_batch_files)} US podcast batch files")
    
    show_episode_counts = defaultdict(int)
    
    for blob in us_batch_files:
        try:
            show_name, orientation = extract_show_and_orientation(blob.name)
            
            content = blob.download_as_string()
            episodes = json.loads(content)
            file_counts['us'] += 1
            
            for episode in episodes:
                source_file = episode.get('metadata', {}).get('source_file', '')
                
                # Check for duplicates
                if source_file in seen_source_files:
                    duplicate_count += 1
                    continue
                
                seen_source_files.add(source_file)
                
                # Extract influence score
                influence_score = episode.get('influence_metrics', {}).get('influence_ratio')
                if influence_score is None:
                    continue
                
                # Add episode
                all_episodes.append({
                    'influence_score': float(influence_score),
                    'orientation': orientation,
                    'show_name': show_name,
                    'source_file': source_file,
                    'word_count': episode['metadata']['total_word_count'],
                    'causal_density': episode['influence_metrics']['causal_density_per_1k'],
                    'persuasive_density': episode['influence_metrics']['persuasive_density_per_1k']
                })
                episode_counts[orientation] += 1
                show_episode_counts[show_name] += 1
                
        except Exception as e:
            print(f"   Error processing {blob.name}: {e}")
    
    # Print detailed breakdown
    print(f"\n3. Data Loading Summary:")
    print(f"   Total batch files processed: {file_counts['tenet'] + file_counts['us']}")
    print(f"   Total unique episodes: {len(all_episodes)}")
    print(f"   Duplicates skipped: {duplicate_count}")
    print(f"\n   Episodes by orientation:")
    for orientation, count in sorted(episode_counts.items()):
        print(f"     {orientation}: {count}")
    
    print(f"\n   Top 10 shows by episode count:")
    for show, count in sorted(show_episode_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     {show}: {count}")
    
    return all_episodes

def calculate_bootstrap_statistics(data, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals with full documentation."""
    if not data or len(data) == 0:
        return None
    
    scores = np.array(data)
    n = len(scores)
    
    # Original statistics
    original_mean = np.mean(scores)
    original_std = np.std(scores, ddof=1)
    original_median = np.median(scores)
    
    # Bootstrap resampling
    bootstrap_means = []
    bootstrap_stds = []
    bootstrap_medians = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
        bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))
        bootstrap_medians.append(np.median(bootstrap_sample))
    
    # Calculate confidence intervals (percentile method)
    results = {
        'n': n,
        'mean': float(original_mean),
        'std': float(original_std),
        'median': float(original_median),
        'ci_mean': [float(np.percentile(bootstrap_means, 2.5)), 
                    float(np.percentile(bootstrap_means, 97.5))],
        'ci_std': [float(np.percentile(bootstrap_stds, 2.5)), 
                   float(np.percentile(bootstrap_stds, 97.5))],
        'ci_median': [float(np.percentile(bootstrap_medians, 2.5)), 
                      float(np.percentile(bootstrap_medians, 97.5))],
        'bootstrap_distribution': {
            'means': bootstrap_means[:100],  # Save first 100 for verification
            'mean_of_bootstrap_means': float(np.mean(bootstrap_means)),
            'std_of_bootstrap_means': float(np.std(bootstrap_means))
        }
    }
    
    return results

def perform_comprehensive_analysis(episodes):
    """Perform comprehensive bootstrap analysis with full documentation."""
    print("\n📊 PERFORMING BOOTSTRAP ANALYSIS")
    print("=" * 50)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(episodes)
    
    # Group data
    tenet_scores = df[df['orientation'] == 'Tenet']['influence_score'].tolist()
    left_scores = df[df['orientation'] == 'Left wing']['influence_score'].tolist()
    right_scores = df[df['orientation'] == 'Right wing']['influence_score'].tolist()
    
    print(f"\nData groups:")
    print(f"  Tenet: {len(tenet_scores)} episodes")
    print(f"  Left wing: {len(left_scores)} episodes")
    print(f"  Right wing: {len(right_scores)} episodes")
    
    # Calculate bootstrap statistics for each group
    print(f"\nCalculating {N_BOOTSTRAP} bootstrap iterations...")
    
    results = {
        'data_source': 'enhanced_analysis_files',
        'calculation_method': 'influence_ratio = (causal_count + persuasive_count) / word_count * 1000',
        'bootstrap_iterations': N_BOOTSTRAP,
        'random_seed': RANDOM_SEED,
        'timestamp': datetime.now().isoformat(),
        'groups': {}
    }
    
    # Process each group
    for group_name, group_data in [('tenet', tenet_scores), 
                                   ('left_wing', left_scores), 
                                   ('right_wing', right_scores)]:
        if group_data:
            print(f"\n  Processing {group_name}...")
            stats = calculate_bootstrap_statistics(group_data, N_BOOTSTRAP)
            results['groups'][group_name] = stats
            
            print(f"    Mean: {stats['mean']:.3f} (95% CI: [{stats['ci_mean'][0]:.3f}, {stats['ci_mean'][1]:.3f}])")
    
    # Calculate differences between groups
    print("\nCalculating group differences...")
    
    # Right vs Left difference
    if right_scores and left_scores:
        diff_scores = []
        for _ in range(N_BOOTSTRAP):
            right_sample = np.random.choice(right_scores, size=len(right_scores), replace=True)
            left_sample = np.random.choice(left_scores, size=len(left_scores), replace=True)
            diff_scores.append(np.mean(right_sample) - np.mean(left_sample))
        
        results['right_vs_left_difference'] = {
            'mean': np.mean(right_scores) - np.mean(left_scores),
            'ci': [float(np.percentile(diff_scores, 2.5)), 
                   float(np.percentile(diff_scores, 97.5))]
        }
        print(f"  Right - Left: {results['right_vs_left_difference']['mean']:.3f} "
              f"(95% CI: {results['right_vs_left_difference']['ci']})")
    
    # Calculate amplification factors relative to Tenet
    if tenet_scores:
        tenet_mean = np.mean(tenet_scores)
        
        # Right wing amplification
        if right_scores:
            amp_scores = []
            for _ in range(N_BOOTSTRAP):
                right_sample = np.random.choice(right_scores, size=len(right_scores), replace=True)
                amp_scores.append(np.mean(right_sample) / tenet_mean)
            
            results['right_amplification'] = {
                'factor': np.mean(right_scores) / tenet_mean,
                'ci': [float(np.percentile(amp_scores, 2.5)), 
                       float(np.percentile(amp_scores, 97.5))]
            }
            print(f"  Right/Tenet: {results['right_amplification']['factor']:.3f}× "
                  f"(95% CI: {results['right_amplification']['ci']})")
        
        # Left wing amplification
        if left_scores:
            amp_scores = []
            for _ in range(N_BOOTSTRAP):
                left_sample = np.random.choice(left_scores, size=len(left_scores), replace=True)
                amp_scores.append(np.mean(left_sample) / tenet_mean)
            
            results['left_amplification'] = {
                'factor': np.mean(left_scores) / tenet_mean,
                'ci': [float(np.percentile(amp_scores, 2.5)), 
                       float(np.percentile(amp_scores, 97.5))]
            }
            print(f"  Left/Tenet: {results['left_amplification']['factor']:.3f}× "
                  f"(95% CI: {results['left_amplification']['ci']})")
    
    return results

def save_results_with_documentation(results, episodes):
    """Save results with complete documentation."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comprehensive output
    output = {
        'metadata': {
            'analysis_name': 'Bootstrap Analysis of Enhanced Episode Data',
            'timestamp': timestamp,
            'data_source': 'Enhanced analysis files from GCS',
            'total_episodes': len(episodes),
            'calculation_documentation': document_data_source().replace('\n', ' '),
            'influence_score_formula': 'influence_ratio = (causal_count + persuasive_count) / word_count * 1000',
            'bootstrap_method': 'Percentile method with n=1000 iterations',
            'random_seed': RANDOM_SEED
        },
        'results': results,
        'validation': {
            'episode_count_by_orientation': {},
            'influence_score_range': {
                'min': min(ep['influence_score'] for ep in episodes),
                'max': max(ep['influence_score'] for ep in episodes),
                'mean': np.mean([ep['influence_score'] for ep in episodes])
            },
            'sample_episodes': episodes[:5]  # Include sample for verification
        }
    }
    
    # Add episode counts
    for ep in episodes:
        ori = ep['orientation']
        output['validation']['episode_count_by_orientation'][ori] = \
            output['validation']['episode_count_by_orientation'].get(ori, 0) + 1
    
    # Save JSON
    json_filename = f'bootstrap_enhanced_transparent_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved detailed results to {json_filename}")
    
    # Create methodology text for dissertation
    methodology_text = f"""
BOOTSTRAP CONFIDENCE INTERVALS FROM ENHANCED ANALYSIS DATA
==========================================================

Data Source: Enhanced analysis files containing pre-calculated influence scores
Analysis Date: {timestamp}
Random Seed: {RANDOM_SEED} (for reproducibility)

Episode Counts:
  Tenet Media: n = {output['validation']['episode_count_by_orientation'].get('Tenet', 0)}
  US Left-wing: n = {output['validation']['episode_count_by_orientation'].get('Left wing', 0)}
  US Right-wing: n = {output['validation']['episode_count_by_orientation'].get('Right wing', 0)}
  Total: n = {len(episodes)}

Influence Score Calculation:
  Formula: (causal_count + persuasive_count) / word_count × 1000
  Where:
    - causal_count = regex-matched causal patterns
    - persuasive_count = keyword-matched persuasive techniques
    - Normalized per 1000 words

Bootstrap Results (1000 iterations):
"""
    
    # Add group results
    for group_name, group_stats in results['groups'].items():
        group_label = {
            'tenet': 'Tenet Media',
            'left_wing': 'US Left-wing',
            'right_wing': 'US Right-wing'
        }.get(group_name, group_name)
        
        methodology_text += f"\n{group_label}:"
        methodology_text += f"\n  Mean: {group_stats['mean']:.3f} (95% CI: [{group_stats['ci_mean'][0]:.3f}, {group_stats['ci_mean'][1]:.3f}])"
        methodology_text += f"\n  n = {group_stats['n']}"
    
    # Add differences
    if 'right_vs_left_difference' in results:
        diff = results['right_vs_left_difference']
        methodology_text += f"\n\nRight vs Left difference: {diff['mean']:.3f} "
        methodology_text += f"(95% CI: [{diff['ci'][0]:.3f}, {diff['ci'][1]:.3f}])"
    
    # Add amplification factors
    methodology_text += "\n\nAmplification Factors (relative to Tenet baseline):"
    if 'right_amplification' in results:
        amp = results['right_amplification']
        methodology_text += f"\n  Right-wing: {amp['factor']:.3f}× (95% CI: [{amp['ci'][0]:.3f}×, {amp['ci'][1]:.3f}×])"
    if 'left_amplification' in results:
        amp = results['left_amplification']
        methodology_text += f"\n  Left-wing: {amp['factor']:.3f}× (95% CI: [{amp['ci'][0]:.3f}×, {amp['ci'][1]:.3f}×])"
    
    methodology_text += "\n\nAll confidence intervals calculated using percentile bootstrap method."
    
    # Save methodology text
    text_filename = f'bootstrap_methodology_transparent_{timestamp}.txt'
    with open(text_filename, 'w') as f:
        f.write(methodology_text)
    print(f"💾 Saved methodology text to {text_filename}")
    
    # Upload to GCS
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        for filename in [json_filename, text_filename]:
            blob = bucket.blob(f'Testing_output/{filename}')
            blob.upload_from_filename(filename)
            print(f"☁️  Uploaded {filename} to {destination(BUCKET_NAME, 'Testing_output/' + filename)}")
    except Exception as e:
        print(f"⚠️  Failed to upload to GCS: {e}")
    
    return json_filename, text_filename

def main():
    """Main execution with full transparency."""
    print("=" * 70)
    print("TRANSPARENT BOOTSTRAP ANALYSIS FROM ENHANCED DATA")
    print("=" * 70)
    
    # Document data source
    document_data_source()
    
    # Load data with validation
    episodes = load_all_episodes_with_validation()
    
    if not episodes:
        print("❌ No episodes loaded! Check data source.")
        return
    
    # Perform analysis
    results = perform_comprehensive_analysis(episodes)
    
    # Save results with documentation
    json_file, text_file = save_results_with_documentation(results, episodes)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey findings from enhanced analysis data:")
    
    # Print summary
    if 'groups' in results:
        for group_name in ['tenet', 'left_wing', 'right_wing']:
            if group_name in results['groups']:
                stats = results['groups'][group_name]
                label = {'tenet': 'Tenet', 'left_wing': 'Left', 'right_wing': 'Right'}[group_name]
                print(f"  {label}: {stats['mean']:.3f} (95% CI: [{stats['ci_mean'][0]:.3f}, {stats['ci_mean'][1]:.3f}])")
    
    print(f"\nThese results are based on the influence scores calculated in your")
    print(f"enhanced analysis pipeline, which uses the formula:")
    print(f"  influence_ratio = (causal_count + persuasive_count) / word_count × 1000")
    print(f"\nThis represents the density of influence techniques per 1000 words.")

if __name__ == "__main__":
    main()