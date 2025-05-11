#!/usr/bin/env python3
"""
Compare all reinforcement learning algorithms in the project.
This script generates comprehensive comparison visualizations.
"""
import os
import argparse
import glob
from utils.visualization import generate_comparison_plots

def main():
    parser = argparse.ArgumentParser(description="Compare RL algorithms for CARLA autonomous driving")
    parser.add_argument("--algorithms", nargs="+", 
                      default=["dqn", "ddqn", "dueling_dqn", "sarsa", "ppo"],
                      help="List of algorithms to compare")
    parser.add_argument("--results_dir", default="results",
                      help="Directory containing algorithm results")
    parser.add_argument("--output_dir", default="comparison_plots",
                      help="Directory to save comparison plots")
    parser.add_argument("--episode_count", type=int, default=None,
                      help="Limit comparison to the first N episodes")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find results directories for each algorithm
    run_dirs = {}
    for algo in args.algorithms:
        # Try to find the most recent run for each algorithm
        algo_dirs = glob.glob(os.path.join(args.results_dir, f"{algo}*"))
        if not algo_dirs:
            print(f"Warning: No results found for algorithm {algo}")
            continue
        
        # Use the most recent directory (assuming naming includes timestamp)
        most_recent = max(algo_dirs, key=os.path.getctime)
        run_dirs[algo] = most_recent
        print(f"Using {most_recent} for {algo}")
    
    if not run_dirs:
        print("Error: No algorithm results found.")
        return
    
    print(f"Comparing algorithms: {', '.join(run_dirs.keys())}")
    print(f"Generating plots in: {args.output_dir}")
    
    # Generate comparison plots
    generate_comparison_plots(list(run_dirs.keys()), run_dirs, args.output_dir)
    
    # Generate summary report
    generate_summary_report(run_dirs, args.output_dir)
    
    print("Comparison complete! Check the output directory for results.")

def generate_summary_report(run_dirs, output_dir):
    """
    Generate a summary report of the comparison.
    
    Args:
        run_dirs (dict): Dictionary mapping algorithm names to run directories
        output_dir (str): Directory to save the report
    """
    from utils.visualization import load_metrics
    import numpy as np
    
    report_path = os.path.join(output_dir, "comparison_summary.md")
    
    with open(report_path, 'w') as f:
        f.write("# Reinforcement Learning Algorithm Comparison Summary\n\n")
        
        # Get metrics for each algorithm
        algorithm_metrics = {}
        for algo, run_dir in run_dirs.items():
            metrics_path = os.path.join(run_dir, f"{algo}_metrics.json")
            metrics = load_metrics(metrics_path)
            algorithm_metrics[algo] = metrics
        
        # Compare final performance (last 10 episodes)
        f.write("## Final Performance (Avg. of Last 10 Episodes)\n\n")
        f.write("| Algorithm | Reward | Success Rate | Collision Rate | Lane Invasion Rate |\n")
        f.write("|-----------|--------|--------------|----------------|-------------------|\n")
        
        for algo, metrics in algorithm_metrics.items():
            # Calculate average reward from last 10 episodes
            if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
                last_n = min(10, len(metrics['episode_rewards']))
                final_reward = np.mean(metrics['episode_rewards'][-last_n:])
            else:
                final_reward = "N/A"
                
            # Calculate success rate
            if 'episode_stats' in metrics and 'success_rate' in metrics['episode_stats']:
                success_rate = np.mean(metrics['episode_stats']['success_rate'][-10:]) * 100
                success_rate_str = f"{success_rate:.1f}%"
            else:
                success_rate_str = "N/A"
                
            # Calculate collision rate
            if 'episode_stats' in metrics and 'collision_count' in metrics['episode_stats']:
                collision_rate = np.mean(metrics['episode_stats']['collision_count'][-10:])
                collision_rate_str = f"{collision_rate:.2f}"
            else:
                collision_rate_str = "N/A"
                
            # Calculate lane invasion rate
            if 'episode_stats' in metrics and 'lane_invasion_count' in metrics['episode_stats']:
                invasion_rate = np.mean(metrics['episode_stats']['lane_invasion_count'][-10:])
                invasion_rate_str = f"{invasion_rate:.2f}"
            else:
                invasion_rate_str = "N/A"
                
            f.write(f"| {algo.upper()} | {final_reward:.2f} | {success_rate_str} | {collision_rate_str} | {invasion_rate_str} |\n")
        
        f.write("\n## Junction Performance\n\n")
        f.write("| Algorithm | Left Turns | Right Turns | Forward | Junction Success |\n")
        f.write("|-----------|------------|-------------|---------|------------------|\n")
        
        for algo, metrics in algorithm_metrics.items():
            # Junction actions
            if 'junction_actions' in metrics:
                left = metrics['junction_actions'].get('left', 0)
                right = metrics['junction_actions'].get('right', 0)
                forward = metrics['junction_actions'].get('forward', 0)
            else:
                left, right, forward = "N/A", "N/A", "N/A"
                
            # Junction success rate
            if 'episode_stats' in metrics and 'junction_success_rate' in metrics['episode_stats']:
                junction_success = np.mean(metrics['episode_stats']['junction_success_rate'][-10:]) * 100
                junction_success_str = f"{junction_success:.1f}%"
            else:
                junction_success_str = "N/A"
                
            f.write(f"| {algo.upper()} | {left} | {right} | {forward} | {junction_success_str} |\n")
            
        f.write("\n## Training Efficiency\n\n")
        f.write("| Algorithm | Episodes | Total Steps | Avg. Steps/Episode | Training Time (h) |\n")
        f.write("|-----------|----------|-------------|--------------------|-----------------|\n")
        
        for algo, metrics in algorithm_metrics.items():
            # Episodes and steps
            if 'episode_rewards' in metrics:
                episodes = len(metrics['episode_rewards'])
            else:
                episodes = "N/A"
                
            if 'episode_lengths' in metrics:
                total_steps = sum(metrics['episode_lengths'])
                avg_steps = total_steps / len(metrics['episode_lengths']) if len(metrics['episode_lengths']) > 0 else 0
            else:
                total_steps, avg_steps = "N/A", "N/A"
                
            # Training time
            if 'computational_metrics' in metrics and 'training_time_per_episode' in metrics['computational_metrics']:
                times = metrics['computational_metrics']['training_time_per_episode']
                total_time_hours = sum(times) / 3600 if times else 0
            else:
                total_time_hours = "N/A"
                
            f.write(f"| {algo.upper()} | {episodes} | {total_steps} | {avg_steps:.1f} | {total_time_hours:.2f} |\n")
            
        f.write("\n## Action Distribution\n\n")
        f.write("| Algorithm | No Action | Throttle | Left | Right | Throttle+Left | Throttle+Right | Brake |\n")
        f.write("|-----------|-----------|----------|------|-------|---------------|----------------|-------|\n")
        
        for algo, metrics in algorithm_metrics.items():
            # Action distribution
            if 'action_distribution' in metrics:
                action_dist = metrics['action_distribution']
                actions = [action_dist.get(str(i), 0) for i in range(7)]
                action_str = " | ".join([str(a) for a in actions])
            else:
                action_str = "N/A | N/A | N/A | N/A | N/A | N/A | N/A"
                
            f.write(f"| {algo.upper()} | {action_str} |\n")
        
        f.write("\n## Plots Generated\n\n")
        f.write("The following plots have been generated in the output directory:\n\n")
        f.write("- **reward_curves.png**: Episode rewards during training\n")
        f.write("- **learning_efficiency.png**: Reward vs. environment steps\n")
        f.write("- **loss_curves.png**: Training loss curves\n")
        f.write("- **centerline_deviation.png**: Average centerline deviation\n")
        f.write("- **collision_rates.png**: Collision rates during training\n")
        f.write("- **action_distribution.png**: Distribution of actions taken\n")
        f.write("- **final_performance.png**: Comparison of final performance metrics\n")
        f.write("- **computational_efficiency.png**: Comparison of computational efficiency\n")
        
        for algo in algorithm_metrics.keys():
            f.write(f"- **{algo}_position_heatmap.png**: Position heatmap for {algo.upper()}\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("Based on the metrics above, the algorithms can be ranked as follows:\n\n")
        
        # Rank algorithms by final reward
        algorithms_ranked = []
        for algo, metrics in algorithm_metrics.items():
            if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
                last_n = min(10, len(metrics['episode_rewards']))
                final_reward = np.mean(metrics['episode_rewards'][-last_n:])
                algorithms_ranked.append((algo, final_reward))
        
        # Sort by final reward (descending)
        algorithms_ranked.sort(key=lambda x: x[1], reverse=True)
        
        f.write("1. **Best Overall Performance**:\n")
        for i, (algo, reward) in enumerate(algorithms_ranked):
            f.write(f"   {i+1}. {algo.upper()} (Avg. Reward: {reward:.2f})\n")
        
        f.write("\nThis ranking is based solely on the final average reward. For a complete analysis, consider all metrics displayed in the plots.\n")
    
    print(f"Generated summary report: {report_path}")

if __name__ == "__main__":
    main() 