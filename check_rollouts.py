#!/usr/bin/env python3
"""
Quick rollout validation tool for Harbor training.

Categorizes trials into:
- Legitimate failures (agent tried but failed task)
- System issues (crashes, timeouts, no responses)
- In progress
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def check_trial(trial_dir: Path) -> dict:
    """Analyze a single trial and return status."""
    result = {
        "name": trial_dir.name,
        "status": "unknown",
        "reward": None,
        "episodes": 0,
        "turns": 0,
        "duration": 0,
        "error": None,
        "total_tokens": 0,
        "avg_completion_tokens": 0,
    }
    
    agent_dir = trial_dir / "agent"
    verifier_dir = trial_dir / "verifier"
    config_file = trial_dir / "config.json"
    reward_file = verifier_dir / "reward.txt"
    
    # Check if agent directory exists
    if not agent_dir.exists():
        result["status"] = "no_agent"
        result["error"] = "Agent directory missing"
        return result
    
    # Count episodes and turns
    episodes = list(agent_dir.glob("episode-*"))
    result["episodes"] = len(episodes)
    
    # Count turns and estimate tokens
    response_files = list(agent_dir.glob("episode-*/response.txt"))
    result["turns"] = len(response_files)
    
    # Estimate token counts from response files
    total_chars = 0
    for resp_file in response_files:
        if resp_file.exists():
            try:
                total_chars += resp_file.stat().st_size
            except:
                pass
    
    # Rough estimate: ~4 chars per token for responses
    result["total_tokens"] = total_chars // 4 if total_chars > 0 else 0
    result["avg_completion_tokens"] = (total_chars // 4) // len(response_files) if response_files else 0
    
    # Get timing
    prompt_file = agent_dir / "episode-0" / "prompt.txt"
    if prompt_file.exists():
        start_time = prompt_file.stat().st_mtime
        if reward_file.exists():
            end_time = reward_file.stat().st_mtime
            result["duration"] = end_time - start_time
        else:
            # Trial still running or failed
            current_time = datetime.now().timestamp()
            result["duration"] = current_time - start_time
    
    # Check reward
    if reward_file.exists():
        result["reward"] = int(reward_file.read_text().strip())
        
        # Legitimate completion (passed or failed)
        if result["turns"] > 0:
            if result["reward"] == 1:
                result["status"] = "passed"
            else:
                result["status"] = "failed"
        else:
            result["status"] = "no_turns"
            result["error"] = "Completed but no agent responses"
    else:
        # No reward file - check why
        if result["episodes"] == 0:
            result["status"] = "no_episodes"
            result["error"] = "No episodes created"
        elif result["turns"] == 0:
            result["status"] = "in_progress"
            result["error"] = "Agent generating first response"
        elif result["duration"] > 600:  # 10 minutes
            result["status"] = "timeout"
            result["error"] = f"Running for {result['duration']:.0f}s with no completion"
        else:
            result["status"] = "in_progress"
    
    # Check for Docker/system issues
    if result["status"] in ["failed", "no_turns"]:
        # Check if agent actually ran
        episode_0 = agent_dir / "episode-0"
        if episode_0.exists():
            response_file = episode_0 / "response.txt"
            if not response_file.exists():
                result["status"] = "agent_crash"
                result["error"] = "Agent didn't generate response"
            elif response_file.stat().st_size == 0:
                result["status"] = "empty_response"
                result["error"] = "Agent generated empty response"
    
    return result


def print_summary(trials: list[dict], verbose: bool = False, show_stats: bool = False):
    """Print analysis summary."""
    # Categorize
    categories = defaultdict(list)
    task_stats = defaultdict(lambda: {"passed": 0, "failed": 0})
    
    for trial in trials:
        categories[trial["status"]].append(trial)
        
        # Track per-task stats
        if trial["status"] in ["passed", "failed"]:
            task_name = trial["name"].split("__")[0]  # Extract task ID
            task_stats[task_name][trial["status"]] += 1
    
    total = len(trials)
    
    print("=" * 80)
    print(f"HARBOR ROLLOUT VALIDATION ({total} trials)")
    print("=" * 80)
    print()
    
    # Success summary
    passed = len(categories["passed"])
    failed = len(categories["failed"])
    completed = passed + failed
    
    if completed > 0:
        print(f"✅ COMPLETED: {completed} ({completed/total*100:.1f}%)")
        print(f"   - Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"   - Failed legitimately: {failed} ({failed/total*100:.1f}%)")
        
        # Stats for passed trials
        if passed > 0:
            passed_durations = [t["duration"] for t in categories["passed"]]
            passed_turns = [t["turns"] for t in categories["passed"]]
            passed_tokens = [t["total_tokens"] for t in categories["passed"]]
            avg_duration = sum(passed_durations) / passed
            avg_turns = sum(passed_turns) / passed
            avg_tokens = sum(passed_tokens) / passed
            min_duration = min(passed_durations)
            max_duration = max(passed_durations)
            print(f"   - Success duration: avg={avg_duration:.1f}s, min={min_duration:.1f}s, max={max_duration:.1f}s")
            print(f"   - Success turns: avg={avg_turns:.1f}, min={min(passed_turns)}, max={max(passed_turns)}")
            print(f"   - Success tokens: avg={avg_tokens:.0f}, min={min(passed_tokens)}, max={max(passed_tokens)}")
        
        # Stats for failed trials
        if failed > 0:
            failed_durations = [t["duration"] for t in categories["failed"]]
            failed_turns = [t["turns"] for t in categories["failed"]]
            failed_tokens = [t["total_tokens"] for t in categories["failed"]]
            avg_duration = sum(failed_durations) / failed
            avg_turns = sum(failed_turns) / failed
            avg_tokens = sum(failed_tokens) / failed
            print(f"   - Failed duration: avg={avg_duration:.1f}s")
            print(f"   - Failed turns: avg={avg_turns:.1f}, min={min(failed_turns)}, max={max(failed_turns)}")
            print(f"   - Failed tokens: avg={avg_tokens:.0f}, min={min(failed_tokens)}, max={max(failed_tokens)}")
        
        print()
    
    # Issues
    issues = [
        ("agent_crash", "Agent crashes (no response generated)"),
        ("empty_response", "Empty responses"),
        ("no_turns", "Completed but no turns"),
        ("no_episodes", "No episodes created"),
        ("timeout", "Timeouts (>10min)"),
    ]
    
    issue_count = sum(len(categories[cat]) for cat, _ in issues)
    
    if issue_count > 0:
        print(f"⚠️  ISSUES DETECTED: {issue_count} ({issue_count/total*100:.1f}%)")
        for category, description in issues:
            count = len(categories[category])
            if count > 0:
                print(f"   - {description}: {count}")
                if verbose:
                    for trial in categories[category][:3]:
                        print(f"     • {trial['name']}: {trial['error']}")
        print()
    
    # In progress
    in_progress = len(categories["in_progress"])
    if in_progress > 0:
        print(f"🔄 IN PROGRESS: {in_progress} ({in_progress/total*100:.1f}%)")
        print()
    
    # Health check
    print("=" * 80)
    if issue_count == 0 and completed > 0:
        print("✅ SYSTEM HEALTHY - All rollouts completing normally")
    elif issue_count / total < 0.1:  # Less than 10% issues
        print("⚠️  MOSTLY HEALTHY - Some issues but majority working")
    else:
        print("❌ ISSUES DETECTED - Significant problems with rollouts")
    print("=" * 80)
    print()
    
    # Additional statistics
    if show_stats and completed > 0:
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        
        # Task difficulty distribution
        if task_stats:
            print("\n📊 TASK PERFORMANCE (attempts on each task):")
            print("-" * 80)
            print(f"{'Task ID':<12} | {'Attempts':>8} | {'Passed':>6} | {'Failed':>6} | {'Success Rate':>12} | {'Zero Adv?':>10}")
            print("-" * 80)
            
            # Categorize tasks by outcome uniformity (for advantage analysis)
            all_pass_tasks = []
            all_fail_tasks = []
            mixed_tasks = []
            
            # Sort by success rate
            sorted_tasks = sorted(
                task_stats.items(),
                key=lambda x: x[1]["passed"] / (x[1]["passed"] + x[1]["failed"]) if (x[1]["passed"] + x[1]["failed"]) > 0 else 0,
                reverse=True
            )
            
            for task_id, stats in sorted_tasks[:15]:  # Show top 15
                attempts = stats["passed"] + stats["failed"]
                success_rate = stats["passed"] / attempts * 100 if attempts > 0 else 0
                
                # Check if all outcomes are the same (causes zero advantage)
                zero_advantage = ""
                if attempts >= 4:  # Only check if we have enough rollouts
                    if stats["passed"] == attempts:
                        zero_advantage = "⚠️  YES"
                        all_pass_tasks.append(task_id)
                    elif stats["failed"] == attempts:
                        zero_advantage = "⚠️  YES"
                        all_fail_tasks.append(task_id)
                    else:
                        mixed_tasks.append(task_id)
                
                print(f"{task_id:<12} | {attempts:>8} | {stats['passed']:>6} | {stats['failed']:>6} | {success_rate:>11.1f}% | {zero_advantage:>10}")
            
            # Zero advantage warning (critical for RL training)
            print()
            print("⚠️  ADVANTAGE ANALYSIS (for RL training):")
            print("-" * 80)
            
            # Count all tasks with enough attempts
            tasks_with_enough_attempts = {
                task_id: stats for task_id, stats in task_stats.items()
                if stats["passed"] + stats["failed"] >= 4
            }
            
            if tasks_with_enough_attempts:
                # Categorize all tasks
                for task_id, stats in tasks_with_enough_attempts.items():
                    attempts = stats["passed"] + stats["failed"]
                    if task_id not in all_pass_tasks and task_id not in all_fail_tasks and task_id not in mixed_tasks:
                        if stats["passed"] == attempts:
                            all_pass_tasks.append(task_id)
                        elif stats["failed"] == attempts:
                            all_fail_tasks.append(task_id)
                        else:
                            mixed_tasks.append(task_id)
                
                total_analyzed = len(all_pass_tasks) + len(all_fail_tasks) + len(mixed_tasks)
                zero_adv_count = len(all_pass_tasks) + len(all_fail_tasks)
                
                print(f"Tasks with 4+ rollouts: {total_analyzed}")
                print(f"  • All rollouts PASSED (100% rate): {len(all_pass_tasks)} tasks")
                print(f"  • All rollouts FAILED (0% rate):   {len(all_fail_tasks)} tasks")
                print(f"  • Mixed outcomes:                   {len(mixed_tasks)} tasks")
                print()
                print(f"⚠️  ZERO ADVANTAGE: {zero_adv_count}/{total_analyzed} tasks ({zero_adv_count/total_analyzed*100:.1f}%)")
                print()
                print("When all rollouts for a task have the SAME outcome, the per-task")
                print("advantage normalization produces zero advantages → zero loss → no learning!")
                print()
                
                if zero_adv_count / total_analyzed > 0.3:
                    print("💡 RECOMMENDATION: Consider one of these fixes:")
                    print("   1. Set 'advantage = false' in config (use raw rewards)")
                    print("   2. Increase temperature for more diverse outcomes per task")
                    print("   3. Use global advantage normalization across batch")
                
                # Show examples
                if len(all_pass_tasks) > 0:
                    print(f"\n   Tasks where ALL pass: {', '.join(all_pass_tasks[:5])}")
                    if len(all_pass_tasks) > 5:
                        print(f"                         (+ {len(all_pass_tasks) - 5} more)")
                if len(all_fail_tasks) > 0:
                    print(f"   Tasks where ALL fail: {', '.join(all_fail_tasks[:5])}")
                    if len(all_fail_tasks) > 5:
                        print(f"                         (+ {len(all_fail_tasks) - 5} more)")
        
        # Duration distribution
        print("\n⏱️  DURATION ANALYSIS:")
        print("-" * 80)
        all_completed = categories["passed"] + categories["failed"]
        if all_completed:
            durations = [t["duration"] for t in all_completed]
            durations.sort()
            
            print(f"Min duration:    {min(durations):6.1f}s")
            print(f"Median duration: {durations[len(durations)//2]:6.1f}s")
            print(f"P90 duration:    {durations[int(len(durations)*0.9)]:6.1f}s")
            print(f"Max duration:    {max(durations):6.1f}s")
        
        # Turns distribution
        print("\n🔄 TURNS ANALYSIS:")
        print("-" * 80)
        if all_completed:
            turns = [t["turns"] for t in all_completed]
            turns_dist = defaultdict(int)
            for t in turns:
                turns_dist[t] += 1
            
            print("Turn count distribution:")
            for turn_count in sorted(turns_dist.keys())[:10]:
                count = turns_dist[turn_count]
                bar = "█" * int(count / max(turns_dist.values()) * 40)
                print(f"  {turn_count:2d} turns: {count:3d} trials {bar}")
        
        # Token distribution
        print("\n🔤 TOKEN ANALYSIS:")
        print("-" * 80)
        if all_completed:
            tokens = [t["total_tokens"] for t in all_completed]
            tokens.sort()
            
            # Check for suspiciously low token counts
            zero_tokens = sum(1 for t in tokens if t == 0)
            low_tokens = sum(1 for t in tokens if 0 < t < 100)
            
            print(f"Total tokens (completion only, estimated):")
            print(f"  Min:    {min(tokens):6d} tokens")
            print(f"  Median: {tokens[len(tokens)//2]:6d} tokens")
            print(f"  P90:    {tokens[int(len(tokens)*0.9)]:6d} tokens")
            print(f"  Max:    {max(tokens):6d} tokens")
            print(f"  Avg:    {sum(tokens)//len(tokens):6d} tokens")
            
            if zero_tokens > 0 or low_tokens > 0:
                print(f"\n  ⚠️  Potential inference issues:")
                if zero_tokens > 0:
                    print(f"    - {zero_tokens} trials with 0 tokens (no generation)")
                if low_tokens > 0:
                    print(f"    - {low_tokens} trials with <100 tokens (very short)")
            
            # Tokens per turn
            avg_tokens_per_turn = [t["total_tokens"] / t["turns"] if t["turns"] > 0 else 0 
                                   for t in all_completed if t["turns"] > 0]
            if avg_tokens_per_turn:
                avg_per_turn = sum(avg_tokens_per_turn) / len(avg_tokens_per_turn)
                print(f"\n  Average tokens per turn: {avg_per_turn:.0f}")
        
        # Recent trend
        print("\n📈 RECENT TREND (chronological order):")
        print("-" * 80)
        completed_sorted = sorted(
            [t for t in trials if t["status"] in ["passed", "failed"]],
            key=lambda x: trials.index(x)
        )
        
        if len(completed_sorted) >= 10:
            # Show last 20 in groups of 5
            for i in range(0, min(20, len(completed_sorted)), 5):
                batch = completed_sorted[i:i+5]
                passed_in_batch = sum(1 for t in batch if t["status"] == "passed")
                print(f"  Trials {i+1:2d}-{min(i+5, len(completed_sorted)):2d}: {passed_in_batch}/5 passed {'✅' if passed_in_batch >= 2 else '⚠️' if passed_in_batch >= 1 else '❌'}")
        
        print()
    
    # Detailed breakdown if verbose
    if verbose and issue_count > 0:
        print("\nDETAILED ISSUE BREAKDOWN:")
        print("-" * 80)
        for category, description in issues:
            trials_with_issue = categories[category]
            if trials_with_issue:
                print(f"\n{description.upper()} ({len(trials_with_issue)}):")
                for trial in trials_with_issue:
                    print(f"  {trial['name'][:30]:30s} | {trial['duration']:6.1f}s | "
                          f"{trial['episodes']} ep | {trial['turns']} turns | {trial['error']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Harbor training rollouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num-trials",
        type=int,
        default=50,
        help="Number of recent trials to check (default: 50)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed breakdown of issues"
    )
    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="Show detailed statistics (task performance, duration, trends)"
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=Path("/tmp/harbor-prime-rl"),
        help="Path to trials directory (default: /tmp/harbor-prime-rl)"
    )
    
    args = parser.parse_args()
    
    trials_dir = args.trials_dir
    
    if not trials_dir.exists():
        print(f"Error: Trials directory not found: {trials_dir}")
        sys.exit(1)
    
    # Get recent trials
    trial_dirs = sorted(
        [d for d in trials_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:args.num_trials]
    
    if not trial_dirs:
        print("No trials found!")
        sys.exit(1)
    
    print(f"Analyzing {len(trial_dirs)} most recent trials...\n")
    
    # Analyze trials
    results = []
    for trial_dir in trial_dirs:
        result = check_trial(trial_dir)
        results.append(result)
    
    # Print summary
    print_summary(results, verbose=args.verbose, show_stats=args.stats)
    
    # Exit code based on health
    issue_count = sum(1 for r in results if r["status"] not in ["passed", "failed", "in_progress"])
    if issue_count > len(results) * 0.2:  # More than 20% issues
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

