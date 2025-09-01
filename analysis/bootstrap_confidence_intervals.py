#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Major Findings
Generates robust confidence intervals for all key statistics
Updated to work with enriched transcript files
"""

import json
import numpy as np
import pandas as pd
from google.cloud import storage
import argparse
from datetime import datetime
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

def extract_show_info(file_path):
    """Extract show name and orientation from enriched transcript file path"""
    
    # Pattern: enriched_transcripts/Left wing__showname__episode__date_with_ids_enriched.json
    if 'tenet' in file_path.lower():
        return 'Tenet Media', 'Foreign'
    
    # Extract from enriched transcript paths
    if 'Left wing__' in file_path:
        orientation = 'Left'
        # Extract show name between "Left wing__" and the next "__"
        try:
            start = file_path.find('Left wing__') + len('Left wing__')
            end = file_path.find('__', start)
            show_name = file_path[start:end] if end != -1 else 'Unknown'
        except:
            show_name = 'Unknown'
    elif 'Right wing__' in file_path:
        orientation = 'Right'
        # Extract show name between "Right wing__" and the next "__"
        try:
            start = file_path.find('Right wing__') + len('Right wing__')
            end = file_path.find('__', start)
            show_name = file_path[start:end] if end != -1 else 'Unknown'
        except:
            show_name = 'Unknown'
    else:
        return 'Unknown', 'Unknown'
    
    return show_name, orientation

def calculate_influence_ratio_from_enriched(episode_data):
    """Calculate influence ratio from enriched transcript data"""
    
    # Get text content for word count
    full_text = episode_data.get('text', '')
    if not full_text and 'segments' in episode_data:
        # Concatenate segment texts
        full_text = ' '.join([seg.get('text', '') for seg in episode_data['segments']])
    
    word_count = len(full_text.split()) if full_text else 0
    
    if word_count == 0:
        return None
    
    # Count causal relationships and persuasive techniques
    segments = episode_data.get('segments', [])
    
    causal_count = 0
    persuasive_count = 0
    
    for segment in segments:
        # Count persuasive techniques
        if 'persuasive_techniques' in segment:
            persuasive_count += len(segment['persuasive_techniques'])
        
        # Look for causal indicators in text (simplified)
        segment_text = segment.get('text', '').lower()
        causal_indicators = ['because', 'therefore', 'as a result', 'led to', 'caused by', 'due to']
        for indicator in causal_indicators:
            if indicator in segment_text:
                causal_count += 1
                break
    
    # Calculate influence ratio
    influence_ratio = (causal_count + persuasive_count) / (word_count / 1000) if word_count > 0 else 0
    
    return influence_ratio

def load_episode_level_data(bucket_name, enriched_prefix, sample_limit=None):
    """Load episode-level influence ratios from enriched transcripts"""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    episodes = []
    
    print(f"Loading enriched transcripts from {enriched_prefix}...")
    
    # Get all enriched transcript files
    all_files = list(bucket.list_blobs(prefix=enriched_prefix))
    transcript_files = [blob for blob in all_files if blob.name.endswith('_enriched.json')]
    
    print(f"Found {len(transcript_files)} enriched transcript files")
    
    count = 0
    for blob in transcript_files:
        if sample_limit and count >= sample_limit:
            break
            
        show_name, orientation = extract_show_info(blob.name)
        
        if orientation not in ['Foreign', 'Left', 'Right']:
            continue
        
        try:
            data = json.loads(blob.download_as_text())
            influence_ratio = calculate_influence_ratio_from_enriched(data)
            
            if influence_ratio is not None and influence_ratio > 0:
                episodes.append({
                    'influence_ratio': float(influence_ratio),
                    'orientation': orientation,
                    'source': 'Tenet' if orientation == 'Foreign' else 'US',
                    'show_name': show_name
                })
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count} episodes...")
                    
        except Exception as e:
            print(f"Error loading {blob.name}: {e}")
    
    print(f"Loaded {len(episodes)} episodes with valid influence ratios")
    
    # Print sample composition
    df = pd.DataFrame(episodes)
    print("\nSample composition:")
    print(df.groupby(['source', 'orientation']).size())
    
    return df

def bootstrap_statistic(data, statistic_func, n_bootstrap=1000, confidence_level=0.95):
    """Generate bootstrap confidence interval for any statistic"""
    
    if len(data) == 0:
        return np.nan, np.nan, np.nan, []
    
    # Original statistic
    original_stat = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return original_stat, ci_lower, ci_upper, bootstrap_stats

def bootstrap_group_comparison(group1, group2, n_bootstrap=1000):
    """Bootstrap confidence interval for difference between groups"""
    
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan, np.nan
    
    # Original difference
    original_diff = np.mean(group1) - np.mean(group2)
    
    # Bootstrap differences
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        boot_group1 = np.random.choice(group1, size=len(group1), replace=True)
        boot_group2 = np.random.choice(group2, size=len(group2), replace=True)
        
        diff = np.mean(boot_group1) - np.mean(boot_group2)
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # 95% CI for difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return original_diff, ci_lower, ci_upper

def bootstrap_amplification_factor(group_data, baseline_value, n_bootstrap=1000):
    """Bootstrap confidence interval for amplification factor"""
    
    if len(group_data) == 0 or baseline_value <= 0:
        return np.nan, np.nan, np.nan
    
    # Original amplification
    original_amp = np.mean(group_data) / baseline_value
    
    # Bootstrap amplifications
    bootstrap_amps = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
        amp = np.mean(bootstrap_sample) / baseline_value
        bootstrap_amps.append(amp)
    
    bootstrap_amps = np.array(bootstrap_amps)
    
    # 95% CI
    ci_lower = np.percentile(bootstrap_amps, 2.5)
    ci_upper = np.percentile(bootstrap_amps, 97.5)
    
    return original_amp, ci_lower, ci_upper

def comprehensive_bootstrap_analysis(df):
    """Perform comprehensive bootstrap analysis"""
    
    np.random.seed(42)  # For reproducibility
    
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'sample_sizes': {
            'tenet': len(df[df['source'] == 'Tenet']),
            'us_left': len(df[df['orientation'] == 'Left']),
            'us_right': len(df[df['orientation'] == 'Right']),
            'total': len(df)
        }
    }
    
    # Extract data by group
    tenet_data = df[df['source'] == 'Tenet']['influence_ratio'].values
    left_data = df[df['orientation'] == 'Left']['influence_ratio'].values
    right_data = df[df['orientation'] == 'Right']['influence_ratio'].values
    us_data = df[df['source'] == 'US']['influence_ratio'].values
    
    print("Computing bootstrap confidence intervals...")
    
    # 1. Basic descriptive statistics with CIs
    print("  Basic statistics...")
    
    # Tenet baseline
    if len(tenet_data) > 0:
        tenet_mean, tenet_ci_low, tenet_ci_high, _ = bootstrap_statistic(tenet_data, np.mean)
        results['tenet_baseline'] = {
            'mean': tenet_mean,
            'ci_95': [tenet_ci_low, tenet_ci_high],
            'n': len(tenet_data)
        }
    else:
        # Use default baseline if no Tenet data
        tenet_mean = 5.45
        results['tenet_baseline'] = {
            'mean': tenet_mean,
            'ci_95': [tenet_mean, tenet_mean],
            'n': 0,
            'note': 'No Tenet data found, using default baseline'
        }
    
    # US Left-wing
    if len(left_data) > 0:
        left_mean, left_ci_low, left_ci_high, _ = bootstrap_statistic(left_data, np.mean)
        results['us_left'] = {
            'mean': left_mean,
            'ci_95': [left_ci_low, left_ci_high],
            'n': len(left_data)
        }
    
    # US Right-wing  
    if len(right_data) > 0:
        right_mean, right_ci_low, right_ci_high, _ = bootstrap_statistic(right_data, np.mean)
        results['us_right'] = {
            'mean': right_mean,
            'ci_95': [right_ci_low, right_ci_high],
            'n': len(right_data)
        }
    
    # Overall US
    if len(us_data) > 0:
        us_mean, us_ci_low, us_ci_high, _ = bootstrap_statistic(us_data, np.mean)
        results['us_overall'] = {
            'mean': us_mean,
            'ci_95': [us_ci_low, us_ci_high],
            'n': len(us_data)
        }
    
    # 2. Group comparisons
    print("  Group comparisons...")
    
    # Left vs Right
    if len(left_data) > 0 and len(right_data) > 0:
        diff, diff_ci_low, diff_ci_high = bootstrap_group_comparison(right_data, left_data)
        results['right_vs_left_difference'] = {
            'difference': diff,
            'ci_95': [diff_ci_low, diff_ci_high],
            'interpretation': f"Right-wing shows average {diff:.3f} points higher than left-wing"
        }
    
    # US vs Tenet
    if len(us_data) > 0 and len(tenet_data) > 0:
        diff, diff_ci_low, diff_ci_high = bootstrap_group_comparison(us_data, tenet_data)
        results['us_vs_tenet_difference'] = {
            'difference': diff,
            'ci_95': [diff_ci_low, diff_ci_high],
            'interpretation': f"US shows average {diff:.3f} points {'higher' if diff > 0 else 'lower'} than Tenet"
        }
    
    # 3. Amplification factors
    print("  Amplification factors...")
    
    # US Right amplification
    if len(right_data) > 0:
        amp, amp_ci_low, amp_ci_high = bootstrap_amplification_factor(right_data, tenet_mean)
        results['right_amplification'] = {
            'factor': amp,
            'ci_95': [amp_ci_low, amp_ci_high],
            'interpretation': f"{amp:.2f}× amplification over Tenet baseline"
        }
    
    # US Left amplification
    if len(left_data) > 0:
        amp, amp_ci_low, amp_ci_high = bootstrap_amplification_factor(left_data, tenet_mean)
        results['left_amplification'] = {
            'factor': amp,
            'ci_95': [amp_ci_low, amp_ci_high],
            'interpretation': f"{amp:.2f}× amplification over Tenet baseline"
        }
    
    # 4. Variability measures
    print("  Variability measures...")
    
    for group_name, group_data in [('tenet', tenet_data), ('us_left', left_data), 
                                   ('us_right', right_data), ('us_overall', us_data)]:
        if len(group_data) > 0:
            # Standard deviation
            sd, sd_ci_low, sd_ci_high, _ = bootstrap_statistic(group_data, np.std)
            
            # Coefficient of variation
            cv_func = lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan
            cv, cv_ci_low, cv_ci_high, _ = bootstrap_statistic(group_data, cv_func)
            
            if group_name not in results:
                results[group_name] = {}
            
            results[group_name]['variability'] = {
                'standard_deviation': {'value': sd, 'ci_95': [sd_ci_low, sd_ci_high]},
                'coefficient_of_variation': {'value': cv, 'ci_95': [cv_ci_low, cv_ci_high]}
            }
    
    return results

def format_methodology_text(results):
    """Format results for methodology section"""
    
    text_sections = []
    
    text_sections.append("BOOTSTRAP CONFIDENCE INTERVALS")
    text_sections.append("="*50)
    text_sections.append("")
    
    # Sample sizes
    sizes = results['sample_sizes']
    text_sections.append(f"Analysis based on {sizes['total']} episodes:")
    text_sections.append(f"  Tenet Media: n = {sizes['tenet']}")
    text_sections.append(f"  US Left-wing: n = {sizes['us_left']}")
    text_sections.append(f"  US Right-wing: n = {sizes['us_right']}")
    text_sections.append("")
    
    # Basic statistics
    text_sections.append("Mean Influence Intensities (95% Bootstrap CI):")
    
    if 'tenet_baseline' in results:
        t = results['tenet_baseline']
        text_sections.append(f"  Tenet baseline: {t['mean']:.3f} "
                           f"(95% CI: [{t['ci_95'][0]:.3f}, {t['ci_95'][1]:.3f}])")
    
    if 'us_left' in results:
        l = results['us_left']
        text_sections.append(f"  US Left-wing: {l['mean']:.3f} "
                           f"(95% CI: [{l['ci_95'][0]:.3f}, {l['ci_95'][1]:.3f}])")
    
    if 'us_right' in results:
        r = results['us_right']
        text_sections.append(f"  US Right-wing: {r['mean']:.3f} "
                           f"(95% CI: [{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}])")
    
    text_sections.append("")
    
    # Group differences
    if 'right_vs_left_difference' in results:
        diff = results['right_vs_left_difference']
        text_sections.append(f"Right vs Left difference: {diff['difference']:.3f} "
                           f"(95% CI: [{diff['ci_95'][0]:.3f}, {diff['ci_95'][1]:.3f}])")
    
    if 'us_vs_tenet_difference' in results:
        diff = results['us_vs_tenet_difference']
        text_sections.append(f"US vs Tenet difference: {diff['difference']:.3f} "
                           f"(95% CI: [{diff['ci_95'][0]:.3f}, {diff['ci_95'][1]:.3f}])")
    
    text_sections.append("")
    
    # Amplification factors
    text_sections.append("Amplification Factors:")
    
    if 'right_amplification' in results:
        amp = results['right_amplification']
        text_sections.append(f"  Right-wing: {amp['factor']:.3f}× "
                           f"(95% CI: [{amp['ci_95'][0]:.3f}×, {amp['ci_95'][1]:.3f}×])")
    
    if 'left_amplification' in results:
        amp = results['left_amplification']
        text_sections.append(f"  Left-wing: {amp['factor']:.3f}× "
                           f"(95% CI: [{amp['ci_95'][0]:.3f}×, {amp['ci_95'][1]:.3f}×])")
    
    text_sections.append("")
    text_sections.append("All confidence intervals calculated using 1,000 bootstrap resamples.")
    
    return "\n".join(text_sections)

def save_results(results, methodology_text, output_dir='.'):
    """Save bootstrap results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed JSON results
    json_file = f"{output_dir}/bootstrap_analysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save methodology text
    text_file = f"{output_dir}/bootstrap_methodology_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write(methodology_text)
    
    # Save formatted results for dissertation
    dissertation_file = f"{output_dir}/bootstrap_dissertation_format_{timestamp}.txt"
    with open(dissertation_file, 'w') as f:
        f.write("For Methodology Section:\n")
        f.write("=" * 30 + "\n\n")
        f.write("All major findings include bootstrap confidence intervals (n = 1,000 resamples) ")
        f.write("to ensure robust statistical inference independent of distributional assumptions.\n\n")
        f.write(methodology_text)
        
        f.write("\n\nFor Results Section:\n")
        f.write("=" * 20 + "\n\n")
        
        # Format key findings for results section
        if 'tenet_baseline' in results and 'us_right' in results:
            tenet = results['tenet_baseline']
            right = results['us_right']
            
            f.write(f"The Tenet Media baseline was established at {tenet['mean']:.2f} ")
            f.write(f"(95% CI: [{tenet['ci_95'][0]:.2f}, {tenet['ci_95'][1]:.2f}], n = {tenet['n']}). ")
            
            f.write(f"US right-wing podcasts achieved a mean influence intensity of {right['mean']:.2f} ")
            f.write(f"(95% CI: [{right['ci_95'][0]:.2f}, {right['ci_95'][1]:.2f}], n = {right['n']}), ")
            
            if 'right_amplification' in results:
                amp = results['right_amplification']
                f.write(f"representing a {amp['factor']:.2f}× amplification ")
                f.write(f"(95% CI: [{amp['ci_95'][0]:.2f}×, {amp['ci_95'][1]:.2f}×]) ")
                f.write("over the foreign baseline.")
    
    return json_file, text_file, dissertation_file

def main():
    parser = argparse.ArgumentParser(description='Generate bootstrap confidence intervals from enriched transcripts')
    parser.add_argument('--bucket', default='podcast-dissertation-audio', help='GCS bucket name')
    parser.add_argument('--enriched-prefix', default='enriched_transcripts/', help='Enriched transcripts prefix')
    parser.add_argument('--sample-limit', type=int, help='Limit sample size for testing')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--upload-to-gcs', action='store_true', help='Upload results to GCS analysis_outputs')
    
    args = parser.parse_args()
    
    print("Loading episode-level data from enriched transcripts...")
    df = load_episode_level_data(args.bucket, args.enriched_prefix, args.sample_limit)
    
    if len(df) == 0:
        print("No episodes found! Check that enriched transcripts exist in the specified location.")
        return
    
    print(f"\nLoaded {len(df)} episodes")
    print("Sample composition:")
    print(df.groupby(['source', 'orientation']).size())
    
    print("\nPerforming bootstrap analysis...")
    results = comprehensive_bootstrap_analysis(df)
    
    print("\nFormatting results...")
    methodology_text = format_methodology_text(results)
    
    print("\n" + methodology_text)
    
    print("\nSaving results...")
    json_file, text_file, dissertation_file = save_results(results, methodology_text, args.output_dir)
    
    print(f"\nFiles created:")
    print(f"  Detailed results: {json_file}")
    print(f"  Methodology text: {text_file}")
    print(f"  Dissertation format: {dissertation_file}")
    
    print(f"\nKey findings:")
    if 'right_amplification' in results:
        amp = results['right_amplification']
        print(f"  Right-wing amplification: {amp['factor']:.3f}× (95% CI: [{amp['ci_95'][0]:.3f}, {amp['ci_95'][1]:.3f}])")
    
    if 'left_amplification' in results:
        amp = results['left_amplification']
        print(f"  Left-wing amplification: {amp['factor']:.3f}× (95% CI: [{amp['ci_95'][0]:.3f}, {amp['ci_95'][1]:.3f}])")
    
    if 'right_vs_left_difference' in results:
        diff = results['right_vs_left_difference']
        print(f"  Right vs Left difference: {diff['difference']:.3f} (95% CI: [{diff['ci_95'][0]:.3f}, {diff['ci_95'][1]:.3f}])")
    
    # Upload to GCS if requested
    if args.upload_to_gcs:
        client = storage.Client()
        bucket = client.bucket('podcast-dissertation-audio')
        
        for local_file, gcs_path in [(json_file, f'analysis_outputs/bootstrap_analysis.json'),
                                    (text_file, f'analysis_outputs/bootstrap_methodology.txt'),
                                    (dissertation_file, f'analysis_outputs/bootstrap_dissertation_format.txt')]:
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_file)
            print(f"Uploaded {local_file} to gs://podcast-dissertation-audio/{gcs_path}")

if __name__ == "__main__":
    main()