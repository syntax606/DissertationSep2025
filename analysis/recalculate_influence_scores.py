#!/usr/bin/env python3
"""
Recalculate influence scores from enhanced analysis files with proper deduplication.
Uses the same data source as the original analysis that produced the 5-7 range scores.
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import os
from google.cloud import storage
import argparse

# Configuration
PROJECT_ID = "handy-vortex-459018-g0"
BUCKET_NAME = "podcast-dissertation-audio"
N_BOOTSTRAP = 1000

# Expected totals for validation
EXPECTED_TOTALS = {
    'tenet': 405,
    'us_right': 3738,
    'us_left': 2289,
    'total': 6432
}

def extract_show_name_from_filename(filename):
    """Extract show name and orientation from enhanced analysis filename."""
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
            if parts[0].lower() in ['right', 'left']:
                orientation = f"{parts[0]} {parts[1]}"
                show_name = '_'.join(parts[2:enhanced_idx])
                return show_name, orientation
    except:
        pass
    
    return 'unknown', 'unknown'

def load_all_unique_episodes():
    """Load all unique episodes from enhanced analysis files using source_file as unique key."""
    print("🔄 Loading unique episodes from enhanced analysis files...")
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # Track unique episodes by source_file
    unique_episodes = {}
    duplicate_count = 0
    
    # Load Tenet episodes
    print("\n📥 Loading Tenet Media episodes...")
    tenet_blobs = bucket.list_blobs(prefix='tenet_enhanced_analysis/')
    tenet_batch_count = 0
    
    for blob in tenet_blobs:
        if blob.name.endswith('.json') and 'batch' in blob.name:
            tenet_batch_count += 1
            try:
                content = blob.download_as_string()
                batch_data = json.loads(content)
                
                for episode in batch_data:
                    # Use source_file as unique identifier
                    source_file = episode.get('metadata', {}).get('source_file', '')
                    if source_file and source_file not in unique_episodes:
                        episode['show_name'] = 'tenet_media'
                        episode['orientation'] = 'Tenet'
                        unique_episodes[source_file] = episode
                    elif source_file:
                        duplicate_count += 1
                        
            except Exception as e:
                print(f"⚠️  Error loading {blob.name}: {e}")
    
    tenet_unique = len([e for e in unique_episodes.values() if e['orientation'] == 'Tenet'])
    print(f"✅ Loaded {tenet_unique} unique Tenet episodes from {tenet_batch_count} batch files")
    print(f"   (Skipped {duplicate_count} duplicates)")
    
    # Load US podcast episodes
    print("\n📥 Loading US podcast episodes...")
    us_blobs = bucket.list_blobs(prefix='us_podcast_enhanced_analysis/')
    us_batch_count = 0
    us_duplicate_count = 0
    
    for blob in us_blobs:
        if blob.name.endswith('.json') and 'batch' in blob.name:
            us_batch_count += 1
            try:
                show_name, orientation = extract_show_name_from_filename(blob.name)
                
                content = blob.download_as_string()
                batch_data = json.loads(content)
                
                for episode in batch_data:
                    source_file = episode.get('metadata', {}).get('source_file', '')
                    if source_file and source_file not in unique_episodes:
                        episode['show_name'] = show_name
                        episode['orientation'] = orientation
                        unique_episodes[source_file] = episode
                    elif source_file:
                        us_duplicate_count += 1
                        
            except Exception as e:
                print(f"⚠️  Error loading {blob.name}: {e}")
    
    # Count by orientation
    orientation_counts = defaultdict(int)
    show_counts = defaultdict(int)
    
    for episode in unique_episodes.values():
        orientation_counts[episode['orientation']] += 1
        show_counts[episode['show_name']] += 1
    
    print(f"✅ Loaded {len(unique_episodes) - tenet_unique} unique US episodes from {us_batch_count} batch files")
    print(f"   (Skipped {us_duplicate_count} duplicates)")
    
    # Validation
    print("\n📊 Episode count validation:")
    print(f"  Tenet: {orientation_counts['Tenet']} (expected: {EXPECTED_TOTALS['tenet']})")
    print(f"  Right wing: {orientation_counts.get('Right wing', 0)} (expected: {EXPECTED_TOTALS['us_right']})")
    print(f"  Left wing: {orientation_counts.get('Left wing', 0)} (expected: {EXPECTED_TOTALS['us_left']})")
    print(f"  Total: {len(unique_episodes)} (expected: {EXPECTED_TOTALS['total']})")
    
    return list(unique_episodes.values()), show_counts

def calculate_bootstrap_ci(scores, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals."""
    if not scores or len(scores) == 0:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'std': 0}
    
    scores_array = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(len(scores), size=len(scores), replace=True)
        sample_scores = scores_array[sample_indices]
        bootstrap_means.append(np.mean(sample_scores))
    
    bootstrap_means = np.array(bootstrap_means)
    
    return {
        'mean': float(np.mean(scores)),
        'ci_lower': float(np.percentile(bootstrap_means, 2.5)),
        'ci_upper': float(np.percentile(bootstrap_means, 97.5)),
        'std': float(np.std(scores))
    }

def analyze_episodes(episodes):
    """Analyze episodes and calculate statistics."""
    print("\n📊 Analyzing episodes...")
    
    # Group by show
    show_episodes = defaultdict(list)
    orientation_episodes = defaultdict(list)
    
    for episode in episodes:
        show_name = episode['show_name']
        orientation = episode['orientation']
        influence_score = episode['influence_metrics']['influence_ratio']
        
        show_episodes[show_name].append({
            'influence_score': influence_score,
            'causal_density': episode['influence_metrics']['causal_density_per_1k'],
            'persuasive_density': episode['influence_metrics']['persuasive_density_per_1k'],
            'word_count': episode['metadata']['total_word_count']
        })
        
        orientation_episodes[orientation].append(influence_score)
    
    # Calculate show-level statistics
    show_results = {}
    
    for show_name, episodes in show_episodes.items():
        influence_scores = [ep['influence_score'] for ep in episodes]
        
        # Find orientation for this show
        show_orientation = next((e['orientation'] for e in episodes if e['show_name'] == show_name), 'unknown')
        
        # Calculate bootstrap CIs
        influence_stats = calculate_bootstrap_ci(influence_scores, N_BOOTSTRAP)
        
        # Calculate totals
        total_words = sum(ep['word_count'] for ep in episodes)
        total_causal = sum(ep['causal_density'] * ep['word_count'] / 1000 for ep in episodes)
        total_persuasive = sum(ep['persuasive_density'] * ep['word_count'] / 1000 for ep in episodes)
        
        show_results[show_name] = {
            'orientation': show_orientation,
            'episode_count': len(episodes),
            'total_words': total_words,
            'influence_score': influence_stats,
            'total_causal_instances': int(total_causal),
            'total_persuasive_instances': int(total_persuasive)
        }
    
    # Calculate orientation-level statistics
    orientation_results = {}
    
    # Map orientation names to match bootstrap analysis
    orientation_mapping = {
        'Tenet': 'tenet_baseline',
        'Left wing': 'us_left', 
        'Right wing': 'us_right'
    }
    
    for orientation, scores in orientation_episodes.items():
        stats = calculate_bootstrap_ci(scores, N_BOOTSTRAP)
        mapped_name = orientation_mapping.get(orientation, orientation)
        orientation_results[mapped_name] = {
            'mean': stats['mean'],
            'ci_95': [stats['ci_lower'], stats['ci_upper']],
            'n': len(scores),
            'original_orientation': orientation
        }
    
    # Calculate differences and amplification factors
    if 'us_right' in orientation_results and 'us_left' in orientation_results:
        right_scores = orientation_episodes['Right wing']
        left_scores = orientation_episodes['Left wing']
        
        # Bootstrap the difference
        bootstrap_diffs = []
        for _ in range(N_BOOTSTRAP):
            right_sample = np.random.choice(right_scores, size=len(right_scores), replace=True)
            left_sample = np.random.choice(left_scores, size=len(left_scores), replace=True)
            bootstrap_diffs.append(np.mean(right_sample) - np.mean(left_sample))
        
        orientation_results['right_vs_left_difference'] = {
            'difference': np.mean(right_scores) - np.mean(left_scores),
            'ci_95': [np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)]
        }
    
    # Calculate amplification factors
    if 'tenet_baseline' in orientation_results:
        tenet_mean = orientation_results['tenet_baseline']['mean']
        
        for key, orientation in [('us_right', 'Right wing'), ('us_left', 'Left wing')]:
            if key in orientation_results:
                scores = orientation_episodes[orientation]
                
                # Bootstrap amplification factor
                bootstrap_amps = []
                for _ in range(N_BOOTSTRAP):
                    sample = np.random.choice(scores, size=len(scores), replace=True)
                    bootstrap_amps.append(np.mean(sample) / tenet_mean)
                
                amp_key = key.replace('us_', '') + '_amplification'
                orientation_results[amp_key] = {
                    'factor': np.mean(scores) / tenet_mean,
                    'ci_95': [np.percentile(bootstrap_amps, 2.5), np.percentile(bootstrap_amps, 97.5)]
                }
    
    return show_results, orientation_results

def generate_report(show_results, orientation_results):
    """Generate formatted report matching bootstrap analysis format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort shows by influence score
    sorted_shows = sorted(show_results.items(), 
                         key=lambda x: x[1]['influence_score']['mean'], 
                         reverse=True)
    
    report = []
    
    # Format to match bootstrap_methodology output
    report.append("BOOTSTRAP CONFIDENCE INTERVALS")
    report.append("=" * 50)
    report.append("")
    
    # Sample sizes
    report.append(f"Analysis based on {sum(r['n'] for r in orientation_results.values() if 'n' in r)} episodes:")
    for key, label in [('tenet_baseline', 'Tenet Media'), 
                       ('us_left', 'US Left-wing'), 
                       ('us_right', 'US Right-wing')]:
        if key in orientation_results:
            report.append(f"  {label}: n = {orientation_results[key]['n']}")
    report.append("")
    
    # Mean influence intensities
    report.append("Mean Influence Intensities (95% Bootstrap CI):")
    for key, label in [('tenet_baseline', 'Tenet baseline'),
                       ('us_left', 'US Left-wing'),
                       ('us_right', 'US Right-wing')]:
        if key in orientation_results:
            r = orientation_results[key]
            report.append(f"  {label}: {r['mean']:.3f} (95% CI: [{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}])")
    report.append("")
    
    # Group differences
    if 'right_vs_left_difference' in orientation_results:
        diff = orientation_results['right_vs_left_difference']
        report.append(f"Right vs Left difference: {diff['difference']:.3f} "
                     f"(95% CI: [{diff['ci_95'][0]:.3f}, {diff['ci_95'][1]:.3f}])")
    
    # Calculate US vs Tenet difference if both exist
    if 'tenet_baseline' in orientation_results and 'us_right' in orientation_results and 'us_left' in orientation_results:
        tenet_mean = orientation_results['tenet_baseline']['mean']
        us_mean = (orientation_results['us_right']['mean'] * orientation_results['us_right']['n'] + 
                   orientation_results['us_left']['mean'] * orientation_results['us_left']['n']) / \
                  (orientation_results['us_right']['n'] + orientation_results['us_left']['n'])
        diff = us_mean - tenet_mean
        report.append(f"US vs Tenet difference: {diff:.3f}")
    
    report.append("")
    
    # Amplification factors
    report.append("Amplification Factors:")
    for key, label in [('right_amplification', 'Right-wing'),
                       ('left_amplification', 'Left-wing')]:
        if key in orientation_results:
            amp = orientation_results[key]
            report.append(f"  {label}: {amp['factor']:.3f}× "
                         f"(95% CI: [{amp['ci_95'][0]:.3f}×, {amp['ci_95'][1]:.3f}×])")
    
    report.append("")
    report.append("All confidence intervals calculated using 1,000 bootstrap resamples.")
    
    # Add individual show rankings
    report.append("\n\nINDIVIDUAL SHOW RANKINGS")
    report.append("-" * 80)
    report.append(f"{'Rank':<6} {'Show':<40} {'Orient':<10} {'Score':<10} {'95% CI':<20} {'Episodes':<10}")
    report.append("-" * 80)
    
    for rank, (show_name, stats) in enumerate(sorted_shows, 1):
        influence = stats['influence_score']
        ci_range = f"[{influence['ci_lower']:.2f}, {influence['ci_upper']:.2f}]"
        
        report.append(f"{rank:<6} {show_name[:39]:<40} {stats['orientation'][:9]:<10} "
                     f"{influence['mean']:<10.2f} {ci_range:<20} {stats['episode_count']:<10}")
    
    return '\n'.join(report), timestamp

def save_results(show_results, orientation_results, report, timestamp):
    """Save results in multiple formats."""
    # Save JSON results
    json_output = {
        'metadata': {
            'timestamp': timestamp,
            'n_bootstrap': N_BOOTSTRAP,
            'total_shows': len(show_results),
            'data_source': 'enhanced_analysis_files'
        },
        'orientation_summary': orientation_results,
        'show_results': show_results
    }
    
    json_filename = f'recalculated_influence_scores_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"\n💾 Saved JSON results to {json_filename}")
    
    # Save text report
    report_filename = f'influence_scores_report_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"💾 Saved text report to {report_filename}")
    
    # Save bootstrap methodology format
    methodology_filename = f'bootstrap_methodology_{timestamp}.txt'
    with open(methodology_filename, 'w') as f:
        # Extract just the bootstrap CI section
        lines = report.split('\n')
        for i, line in enumerate(lines):
            f.write(line + '\n')
            if line.startswith("All confidence intervals calculated"):
                break
    print(f"💾 Saved methodology format to {methodology_filename}")
    
    # Upload to GCS
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        for filename in [json_filename, report_filename, methodology_filename]:
            blob = bucket.blob(f'Testing_output/{filename}')
            blob.upload_from_filename(filename)
            print(f"☁️  Uploaded {filename} to gs://{BUCKET_NAME}/Testing_output/{filename}")
            
    except Exception as e:
        print(f"⚠️  Failed to upload to GCS: {e}")

def main():
    parser = argparse.ArgumentParser(description='Recalculate influence scores from enhanced analysis files')
    parser.add_argument('--bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer bootstrap iterations (100)')
    
    args = parser.parse_args()
    
    global N_BOOTSTRAP
    N_BOOTSTRAP = 100 if args.quick else args.bootstrap
    
    print(f"🚀 Starting influence score recalculation with {N_BOOTSTRAP} bootstrap iterations...")
    print("📝 Using enhanced analysis files (same source as original 5-7 range scores)")
    
    # Load all unique episodes
    episodes, show_counts = load_all_unique_episodes()
    
    if not episodes:
        print("❌ No episodes found!")
        return
    
    # Analyze episodes
    show_results, orientation_results = analyze_episodes(episodes)
    
    # Generate report
    report, timestamp = generate_report(show_results, orientation_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(report.split("INDIVIDUAL SHOW RANKINGS")[0])
    
    # Save results
    save_results(show_results, orientation_results, report, timestamp)
    
    print("\n✅ Analysis complete!")
    
    # Print key findings matching bootstrap format
    print("\n🎯 Key findings (matching bootstrap analysis format):")
    if 'tenet_baseline' in orientation_results:
        t = orientation_results['tenet_baseline']
        print(f"  Tenet baseline: {t['mean']:.3f} (95% CI: [{t['ci_95'][0]:.3f}, {t['ci_95'][1]:.3f}])")
    
    if 'right_amplification' in orientation_results:
        amp = orientation_results['right_amplification']
        print(f"  Right-wing amplification: {amp['factor']:.3f}× (95% CI: [{amp['ci_95'][0]:.3f}×, {amp['ci_95'][1]:.3f}×])")

if __name__ == "__main__":
    main()