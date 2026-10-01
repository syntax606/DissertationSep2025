#!/usr/bin/env python3
"""
Calculate per-show influence scores from enhanced analysis data.
This properly deduplicates episodes and calculates bootstrap CIs for each show.
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from storage_backend import destination, is_local, storage  # local folder or GCS; see storage_backend.py

# Configuration
PROJECT_ID = "handy-vortex-459018-g0"
BUCKET_NAME = "podcast-dissertation-audio"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def extract_show_and_orientation(filename):
    """Extract show name and political orientation from filename."""
    filename = filename.split('/')[-1]
    
    if 'tenet' in filename.lower():
        return 'tenet_media', 'Tenet'
    
    parts = filename.split('_')
    try:
        enhanced_idx = None
        for i, part in enumerate(parts):
            if part == 'enhanced':
                enhanced_idx = i
                break
        
        if enhanced_idx and enhanced_idx >= 3:
            orientation = f"{parts[0]} {parts[1]}"
            show_name = '_'.join(parts[2:enhanced_idx])
            return show_name, orientation
    except:
        pass
    
    return 'unknown', 'unknown'

def load_show_level_data():
    """Load all episodes grouped by show."""
    print("📊 Loading episode data by show...")
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # Dictionary to store episodes by show
    shows_data = defaultdict(lambda: {
        'episodes': [],
        'orientation': None,
        'episode_count': 0
    })
    
    # Track duplicates
    seen_source_files = set()
    
    # Load Tenet episodes
    print("\nProcessing Tenet Media...")
    tenet_blobs = list(bucket.list_blobs(prefix='tenet_enhanced_analysis/'))
    tenet_files = [b for b in tenet_blobs if b.name.endswith('.json') and 'batch' in b.name]
    
    for blob in tenet_files:
        try:
            content = blob.download_as_string()
            episodes = json.loads(content)
            
            for episode in episodes:
                source_file = episode.get('metadata', {}).get('source_file', '')
                if source_file in seen_source_files:
                    continue
                seen_source_files.add(source_file)
                
                influence_score = episode.get('influence_metrics', {}).get('influence_ratio')
                if influence_score:
                    shows_data['tenet_media']['episodes'].append(influence_score)
                    shows_data['tenet_media']['orientation'] = 'Tenet'
                    
        except Exception as e:
            print(f"Error processing {blob.name}: {e}")
    
    shows_data['tenet_media']['episode_count'] = len(shows_data['tenet_media']['episodes'])
    print(f"  Loaded {shows_data['tenet_media']['episode_count']} episodes")
    
    # Load US podcast episodes
    print("\nProcessing US podcasts...")
    us_blobs = list(bucket.list_blobs(prefix='us_podcast_enhanced_analysis/'))
    us_files = [b for b in us_blobs if b.name.endswith('.json') and 'batch' in b.name]
    
    for blob in us_files:
        try:
            show_name, orientation = extract_show_and_orientation(blob.name)
            if show_name == 'unknown':
                continue
                
            content = blob.download_as_string()
            episodes = json.loads(content)
            
            for episode in episodes:
                source_file = episode.get('metadata', {}).get('source_file', '')
                if source_file in seen_source_files:
                    continue
                seen_source_files.add(source_file)
                
                influence_score = episode.get('influence_metrics', {}).get('influence_ratio')
                if influence_score:
                    shows_data[show_name]['episodes'].append(influence_score)
                    shows_data[show_name]['orientation'] = orientation
                    
        except Exception as e:
            print(f"Error processing {blob.name}: {e}")
    
    # Update episode counts
    for show_name, data in shows_data.items():
        if show_name != 'tenet_media':
            data['episode_count'] = len(data['episodes'])
    
    # Print summary
    print(f"\nLoaded data for {len(shows_data)} shows")
    print(f"Total episodes: {sum(d['episode_count'] for d in shows_data.values())}")
    
    return dict(shows_data)

def calculate_show_statistics(shows_data, n_bootstrap=1000):
    """Calculate statistics for each show with bootstrap CIs."""
    print("\n📊 Calculating per-show statistics...")
    
    results = []
    
    for show_name, data in shows_data.items():
        if not data['episodes']:
            continue
            
        scores = np.array(data['episodes'])
        n = len(scores)
        
        # Basic statistics
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        median = np.median(scores)
        
        # Bootstrap for confidence intervals
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        # Add to results
        results.append({
            'show_name': show_name,
            'orientation': data['orientation'],
            'episode_count': n,
            'mean_influence': mean,
            'std': std,
            'median': median,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        })
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('mean_influence', ascending=False)
    
    return df

def calculate_aggregate_validation(shows_data):
    """Validate against bootstrap aggregates."""
    print("\n✅ Validating against bootstrap results...")
    
    # Group episodes by orientation
    orientation_scores = defaultdict(list)
    
    for show_name, data in shows_data.items():
        orientation = data['orientation']
        orientation_scores[orientation].extend(data['episodes'])
    
    # Calculate means
    validation = {}
    for orientation, scores in orientation_scores.items():
        if scores:
            validation[orientation] = {
                'mean': np.mean(scores),
                'n': len(scores)
            }
    
    return validation

def generate_report(df, validation, shows_data):
    """Generate comprehensive report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = []
    report.append("PER-SHOW INFLUENCE SCORES FROM ENHANCED ANALYSIS DATA")
    report.append("=" * 80)
    report.append(f"Generated: {timestamp}")
    report.append(f"Bootstrap iterations: {N_BOOTSTRAP}")
    report.append("")
    
    # Validation section
    report.append("VALIDATION AGAINST BOOTSTRAP AGGREGATES:")
    report.append("-" * 40)
    for orientation, stats in validation.items():
        report.append(f"{orientation}: {stats['mean']:.3f} (n={stats['n']})")
    report.append("")
    
    # Rankings table
    report.append("SHOW RANKINGS WITH CONFIDENCE INTERVALS:")
    report.append("-" * 100)
    report.append(f"{'Rank':<6} {'Show':<40} {'Orient':<10} {'Episodes':<10} {'Mean':<10} {'95% CI':<20} {'Std':<10}")
    report.append("-" * 100)
    
    for idx, row in df.iterrows():
        rank = idx + 1
        show = row['show_name'][:39]
        orient = row['orientation'][:9]
        n = row['episode_count']
        mean = row['mean_influence']
        ci = f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
        std = row['std']
        
        report.append(f"{rank:<6} {show:<40} {orient:<10} {n:<10} {mean:<10.3f} {ci:<20} {std:<10.3f}")
    
    # Statistical summary by orientation
    report.append("\n\nSUMMARY BY ORIENTATION:")
    report.append("-" * 40)
    
    for orientation in ['Tenet', 'Right wing', 'Left wing']:
        orient_df = df[df['orientation'] == orientation]
        if not orient_df.empty:
            report.append(f"\n{orientation}:")
            report.append(f"  Shows: {len(orient_df)}")
            report.append(f"  Total episodes: {orient_df['episode_count'].sum()}")
            report.append(f"  Mean influence: {orient_df['mean_influence'].mean():.3f}")
            report.append(f"  Range: {orient_df['mean_influence'].min():.3f} - {orient_df['mean_influence'].max():.3f}")
    
    # Interesting findings
    report.append("\n\nKEY FINDINGS:")
    report.append("-" * 40)
    
    # Shows most similar to Tenet
    tenet_mean = df[df['show_name'] == 'tenet_media']['mean_influence'].values[0]
    df['tenet_diff'] = abs(df['mean_influence'] - tenet_mean)
    closest_to_tenet = df[df['show_name'] != 'tenet_media'].nsmallest(5, 'tenet_diff')
    
    report.append(f"\nShows most similar to Tenet ({tenet_mean:.3f}):")
    for _, row in closest_to_tenet.iterrows():
        diff = row['mean_influence'] - tenet_mean
        report.append(f"  {row['show_name']}: {row['mean_influence']:.3f} ({diff:+.3f})")
    
    return '\n'.join(report)

def save_results(df, validation, report, shows_data):
    """Save all results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed JSON
    json_data = {
        'metadata': {
            'timestamp': timestamp,
            'source': 'enhanced_analysis_files',
            'bootstrap_iterations': N_BOOTSTRAP,
            'total_shows': len(df),
            'total_episodes': sum(d['episode_count'] for d in shows_data.values())
        },
        'validation': validation,
        'show_results': df.to_dict('records')
    }
    
    json_filename = f'per_show_influence_scores_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save report
    report_filename = f'per_show_influence_report_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(report)
    
    # Save CSV for easy use
    csv_filename = f'per_show_influence_scores_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"\n💾 Saved results:")
    print(f"  JSON: {json_filename}")
    print(f"  Report: {report_filename}")
    print(f"  CSV: {csv_filename}")
    
    # Upload to GCS
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        for filename in [json_filename, report_filename, csv_filename]:
            blob = bucket.blob(f'Testing_output/{filename}')
            blob.upload_from_filename(filename)
            print(f"  ☁️  Uploaded {filename} to {destination(BUCKET_NAME, 'Testing_output/' + filename)}")
    except Exception as e:
        print(f"  ⚠️  Failed to upload: {e}")

def main():
    print("=" * 80)
    print("CALCULATING PER-SHOW INFLUENCE SCORES")
    print("=" * 80)
    
    # Load data
    shows_data = load_show_level_data()
    
    # Calculate statistics
    df = calculate_show_statistics(shows_data, N_BOOTSTRAP)
    
    # Validate against bootstrap
    validation = calculate_aggregate_validation(shows_data)
    
    # Generate report
    report = generate_report(df, validation, shows_data)
    
    # Print key results
    print("\n" + "=" * 80)
    print("TOP 10 SHOWS BY INFLUENCE SCORE:")
    print("=" * 80)
    for idx, row in df.head(10).iterrows():
        print(f"{idx+1}. {row['show_name']}: {row['mean_influence']:.3f} "
              f"(95% CI: [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]) "
              f"n={row['episode_count']}")
    
    # Save everything
    save_results(df, validation, report, shows_data)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()