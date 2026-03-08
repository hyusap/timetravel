from benchmarks.reverse_code_door import ReverseCodeDoorEnv, EpisodeConfig, rewind_policy, rollout_episode

env = ReverseCodeDoorEnv(config=EpisodeConfig(budget=10, failure_penalty=-20, end_episode_on_wrong_unlock=True))
results = [rollout_episode(env, rewind_policy, seed=i) for i in range(100)]

avg_reward = sum(r['total_reward'] for r in results) / len(results)
success_rate = sum(r['success'] for r in results) / len(results)

print(f"avg reward:   {avg_reward:.4f}")
print(f"success rate: {success_rate:.0%}")
print(f"sample rewards: {[round(r['total_reward'], 3) for r in results[:10]]}")
