#!/usr/bin/env python3
"""
Generate comprehensive analysis and comparison of different RL algorithms.
This script generates all visualizations needed for a comparative study.
"""
import os
import argparse
import glob
import json
import numpy as np
from utils.visualization import (
    generate_comparison_plots,
    plot_reward_curves,
    plot_learning_efficiency,
    plot_loss_curves,
    plot_centerline_deviation,
    plot_collision_rates,
    plot_action_distribution,
    plot_position_heatmap,
    plot_final_performance_comparison,
    plot_computational_efficiency
)

def find_algorithm_metrics(results_dir, algorithms=None):
    """
    Find all algorithm metrics files in the results directory.
    
    Args:
        results_dir (str): Directory to search for metrics files
        algorithms (list, optional): List of algorithms to include
    
    Returns:
        dict: Mapping of algorithm names to metrics files
    """
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(results_dir, "**/*_metrics.json"), recursive=True)
    
    if not metrics_files:
        raise ValueError(f"No metrics files found in {results_dir}")
    
    # Group by algorithm
    algorithm_metrics = {}
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'algorithm' in data:
                    algo = data['algorithm'].lower()
                else:
                    # Try to extract from filename
                    filename = os.path.basename(file_path)
                    algo = filename.split('_')[0].lower()
                
                if algorithms and algo not in algorithms:
                    continue
                
                if algo not in algorithm_metrics:
                    algorithm_metrics[algo] = []
                
                algorithm_metrics[algo].append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return algorithm_metrics

def load_metrics_data(algorithm_metrics):
    """
    Load all metrics data from files.
    
    Args:
        algorithm_metrics (dict): Mapping of algorithm names to metrics files
    
    Returns:
        dict: Mapping of algorithm names to metrics data
    """
    data = {}
    
    for algo, file_paths in algorithm_metrics.items():
        # Use the most recent file for each algorithm
        if not file_paths:
            continue
        
        # Sort by timestamp if available
        try:
            most_recent = max(file_paths, key=lambda x: os.path.getmtime(x))
        except:
            most_recent = file_paths[0]
        
        try:
            with open(most_recent, 'r') as f:
                metrics = json.load(f)
                data[algo] = metrics
        except Exception as e:
            print(f"Error loading {most_recent}: {e}")
    
    return data

def generate_algorithm_visualizations(metrics_data, output_dir):
    """
    Generate all visualizations for algorithm comparison.
    
    Args:
        metrics_data (dict): Mapping of algorithm names to metrics data
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_reward_curves(metrics_data, os.path.join(output_dir, 'reward_curves.png'))
    plot_learning_efficiency(metrics_data, os.path.join(output_dir, 'learning_efficiency.png'))
    plot_loss_curves(metrics_data, os.path.join(output_dir, 'loss_curves.png'))
    plot_centerline_deviation(metrics_data, os.path.join(output_dir, 'centerline_deviation.png'))
    plot_collision_rates(metrics_data, os.path.join(output_dir, 'collision_rates.png'))
    plot_action_distribution(metrics_data, os.path.join(output_dir, 'action_distribution.png'))
    
    # Generate individual position heatmaps
    for algo in metrics_data.keys():
        plot_position_heatmap(metrics_data, algo, os.path.join(output_dir, f'{algo}_position_heatmap.png'))
    
    # Generate final performance comparison
    plot_final_performance_comparison(metrics_data, os.path.join(output_dir, 'final_performance.png'))
    
    # Generate computational efficiency comparison
    plot_computational_efficiency(metrics_data, os.path.join(output_dir, 'computational_efficiency.png'))
    
    print(f"Generated all visualizations in {output_dir}")

def generate_summary_report(metrics_data, output_dir):
    """
    Generate a summary report comparing algorithms.
    
    Args:
        metrics_data (dict): Mapping of algorithm names to metrics data
        output_dir (str): Directory to save the report
    """
    report_path = os.path.join(output_dir, 'algorithm_comparison.md')
    
    with open(report_path, 'w') as f:
        f.write("# Reinforcement Learning Algorithm Comparison\n\n")
        
        # Overall performance
        f.write("## Overall Performance\n\n")
        f.write("| Algorithm | Final Reward | Training Steps | Success Rate | Collision Rate |\n")
        f.write("|-----------|--------------|---------------|--------------|---------------|\n")
        
        for algo, data in metrics_data.items():
            # Calculate average reward for last 10 episodes
            final_reward = "N/A"
            if 'episode_rewards' in data and data['episode_rewards']:
                last_n = min(10, len(data['episode_rewards']))
                final_reward = f"{np.mean(data['episode_rewards'][-last_n:]):.2f}"
            
            # Calculate total training steps
            training_steps = "N/A"
            if 'episode_lengths' in data and data['episode_lengths']:
                training_steps = f"{sum(data['episode_lengths'])}"
            
            # Calculate success rate
            success_rate = "N/A"
            if 'episode_stats' in data and 'success_rate' in data['episode_stats']:
                rate = np.mean(data['episode_stats']['success_rate'][-10:]) * 100
                success_rate = f"{rate:.1f}%"
            
            # Calculate collision rate
            collision_rate = "N/A"
            if 'episode_stats' in data and 'collision_count' in data['episode_stats']:
                rate = np.mean(data['episode_stats']['collision_count'][-10:])
                collision_rate = f"{rate:.2f}"
            
            f.write(f"| {algo.upper()} | {final_reward} | {training_steps} | {success_rate} | {collision_rate} |\n")
        
        # Junction navigation
        f.write("\n## Junction Navigation\n\n")
        f.write("| Algorithm | Left Turns | Right Turns | Forward | Success Rate |\n")
        f.write("|-----------|------------|-------------|---------|-------------|\n")
        
        for algo, data in metrics_data.items():
            left = "N/A"
            right = "N/A"
            forward = "N/A"
            if 'junction_actions' in data:
                left = f"{data['junction_actions'].get('left', 0)}"
                right = f"{data['junction_actions'].get('right', 0)}"
                forward = f"{data['junction_actions'].get('forward', 0)}"
            
            success_rate = "N/A"
            if 'episode_stats' in data and 'junction_success_rate' in data['episode_stats']:
                rate = np.mean(data['episode_stats']['junction_success_rate'][-10:]) * 100
                success_rate = f"{rate:.1f}%"
            
            f.write(f"| {algo.upper()} | {left} | {right} | {forward} | {success_rate} |\n")
        
        # Driving quality
        f.write("\n## Driving Quality\n\n")
        f.write("| Algorithm | Avg. Speed | Centerline Deviation | Lane Invasions |\n")
        f.write("|-----------|------------|----------------------|---------------|\n")
        
        for algo, data in metrics_data.items():
            speed = "N/A"
            if 'episode_stats' in data and 'avg_speed' in data['episode_stats']:
                speed = f"{np.mean(data['episode_stats']['avg_speed'][-10:]):.2f} m/s"
            
            deviation = "N/A"
            if 'episode_stats' in data and 'avg_centerline_deviation' in data['episode_stats']:
                deviation = f"{np.mean(data['episode_stats']['avg_centerline_deviation'][-10:]):.2f} m"
            
            invasions = "N/A"
            if 'episode_stats' in data and 'lane_invasion_count' in data['episode_stats']:
                invasions = f"{np.mean(data['episode_stats']['lane_invasion_count'][-10:]):.2f}"
            
            f.write(f"| {algo.upper()} | {speed} | {deviation} | {invasions} |\n")
        
        # Computational efficiency
        f.write("\n## Computational Efficiency\n\n")
        f.write("| Algorithm | Training Time | Inference Time | Memory Usage |\n")
        f.write("|-----------|---------------|----------------|-------------|\n")
        
        for algo, data in metrics_data.items():
            training_time = "N/A"
            inference_time = "N/A"
            memory_usage = "N/A"
            
            if 'computational_metrics' in data:
                comp = data['computational_metrics']
                
                if 'training_time_per_episode' in comp and comp['training_time_per_episode']:
                    training_time = f"{np.mean(comp['training_time_per_episode']):.2f} s/ep"
                
                if 'inference_time_per_step' in comp and comp['inference_time_per_step']:
                    inference_time = f"{np.mean(comp['inference_time_per_step']) * 1000:.2f} ms/step"
                
                if 'memory_usage' in comp and comp['memory_usage']:
                    memory_usage = f"{np.mean(comp['memory_usage']):.2f} GB"
            
            f.write(f"| {algo.upper()} | {training_time} | {inference_time} | {memory_usage} |\n")
        
        # Algorithm ranking
        f.write("\n## Algorithm Ranking\n\n")
        
        # Rank algorithms by final performance
        algo_scores = []
        for algo, data in metrics_data.items():
            score = 0
            
            # Score based on reward
            if 'episode_rewards' in data and data['episode_rewards']:
                score += np.mean(data['episode_rewards'][-10:]) / 10
            
            # Score based on success rate
            if 'episode_stats' in data and 'success_rate' in data['episode_stats']:
                score += np.mean(data['episode_stats']['success_rate'][-10:]) * 50
            
            # Penalty for collisions
            if 'episode_stats' in data and 'collision_count' in data['episode_stats']:
                score -= np.mean(data['episode_stats']['collision_count'][-10:]) * 20
            
            algo_scores.append((algo, score))
        
        # Sort by score
        algo_scores.sort(key=lambda x: x[1], reverse=True)
        
        f.write("Based on combined metrics (reward, success rate, and safety):\n\n")
        for i, (algo, score) in enumerate(algo_scores):
            f.write(f"{i+1}. **{algo.upper()}** (Score: {score:.2f})\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        
        if algo_scores:
            best_algo = algo_scores[0][0].upper()
            f.write(f"The **{best_algo}** algorithm demonstrates the best overall performance in this autonomous driving task. ")
            
            # Add specific strengths of the best algorithm
            if best_algo.lower() in metrics_data:
                data = metrics_data[best_algo.lower()]
                strengths = []
                
                # Check reward
                if 'episode_rewards' in data and data['episode_rewards'] and np.mean(data['episode_rewards'][-10:]) > 0:
                    strengths.append("achieving positive rewards")
                
                # Check success rate
                if 'episode_stats' in data and 'success_rate' in data['episode_stats'] and np.mean(data['episode_stats']['success_rate'][-10:]) > 0.5:
                    strengths.append("high success rate")
                
                # Check collision rate
                if 'episode_stats' in data and 'collision_count' in data['episode_stats'] and np.mean(data['episode_stats']['collision_count'][-10:]) < 1:
                    strengths.append("low collision rate")
                
                if strengths:
                    f.write(f"It excels in {', '.join(strengths)}. ")
            
            # Recommendations
            f.write("\n\nFor future work, consider:\n\n")
            f.write("1. **Hyperparameter optimization** for the best performing algorithms\n")
            f.write("2. **Longer training** to see if performance continues to improve\n")
            f.write("3. **Hybrid approaches** combining the strengths of different algorithms\n")
            f.write("4. **Improving reward design** to better balance task completion and safety\n")
    
    print(f"Generated summary report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive analysis for RL algorithms")
    parser.add_argument("--results_dir", default="results", help="Directory containing algorithm results")
    parser.add_argument("--output_dir", default="analysis", help="Directory to save analysis results")
    parser.add_argument("--algorithms", nargs="+", default=None, help="Specific algorithms to analyze")
    
    args = parser.parse_args()
    
    # Find metrics files for each algorithm
    algorithm_metrics = find_algorithm_metrics(args.results_dir, args.algorithms)
    
    if not algorithm_metrics:
        print(f"No metrics files found for algorithms: {args.algorithms}")
        return
    
    print(f"Found metrics for algorithms: {', '.join(algorithm_metrics.keys())}")
    
    # Load metrics data
    metrics_data = load_metrics_data(algorithm_metrics)
    
    if not metrics_data:
        print("Failed to load any metrics data")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    generate_algorithm_visualizations(metrics_data, args.output_dir)
    
    # Generate summary report
    generate_summary_report(metrics_data, args.output_dir)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 