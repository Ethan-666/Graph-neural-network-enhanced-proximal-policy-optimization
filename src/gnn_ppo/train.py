from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import data_paths, load_config
from .data import validate_prepared_dataset
from .env import PaperAlignedBridgeEnv
from .graph import build_paper_graph_from_config
from .models import PPOAgent
from .reporting import transform_metric
from .utils import ensure_dir, exponential_moving_average, set_seed, timestamp_for_run
from .visualize import plot_reward_loss_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper-aligned masked GNN-PPO dispatcher.")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episode count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--no-depot", action="store_true", help="Disable depot vehicles for this run.")
    parser.add_argument("--run-name", default=None, help="Optional output run directory name.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave unset for auto.")
    return parser.parse_args()


def build_env(config: dict[str, Any]) -> PaperAlignedBridgeEnv:
    paths = data_paths(config)
    validate_prepared_dataset(paths["static_dir"], paths["od_dir"], config["data"]["date"])
    return PaperAlignedBridgeEnv(
        date=config["data"]["date"],
        static_dir=paths["static_dir"],
        od_dir=paths["od_dir"],
        station_features_name=config["data"]["station_features"],
        line_features_name=config["data"]["line_features"],
        departure_info_name=config["data"]["departure_info"],
        env_config=config["env"],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        config["train"]["episodes"] = args.episodes
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.no_depot:
        config["env"]["use_depot"] = False

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = build_env(config)
    paths = data_paths(config)
    graph, feature_builder = build_paper_graph_from_config(
        config,
        static_dir=paths["static_dir"],
        station_features_df=env.station_features_df,
    )
    context_dim = len(env.vehicle_context(next(iter(env.vehicles.values())), datetime.strptime("16:00:00", "%H:%M:%S"))) if env.vehicles else 8
    agent = PPOAgent(
        graph,
        config["model"],
        device,
        action_dim=env.action_catalog.size,
        context_dim=context_dim,
    )

    run_name = args.run_name or timestamp_for_run(seed)
    run_dir = ensure_dir(paths["output_dir"] / run_name)
    config_snapshot = run_dir / "config_used.json"
    config_snapshot.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    env.action_catalog.to_frame().to_csv(run_dir / "action_catalog.csv", index=False)

    rewards_history: list[float] = []
    reward_metro_history: list[float] = []
    reward_bus_history: list[float] = []
    reward_load_history: list[float] = []
    passenger_history: list[int] = []
    additional_bus_waiting_history: list[int] = []
    load_factor_history: list[float] = []
    actor_loss_history: list[float] = []
    critic_loss_history: list[float] = []
    entropy_history: list[float] = []
    clip_frac_history: list[float] = []
    trajectory_buffer: list[dict[str, Any]] = []

    episodes = int(config["train"]["episodes"])
    duration = int(config["data"]["episode_duration_minutes"])
    update_interval = int(config["train"]["update_interval"])

    for episode in tqdm(range(episodes), desc="Training"):
        start_time, _ = env.sample_episode_window(duration)
        try:
            state_vec, context, mask, event = env.reset(start_time, duration)
        except RuntimeError:
            continue
        graph_state = feature_builder.build(state_vec, env.current_time)

        total_reward = 0.0
        total_reward_metro = 0.0
        total_reward_bus = 0.0
        total_reward_load = 0.0
        total_passengers = 0
        total_additional_bus_waiting = 0
        load_factors: list[float] = []
        episode_trajectory: list[dict[str, Any]] = []
        done = False

        while not done:
            action, log_prob, value = agent.choose_action(graph_state, context, mask)
            result = env.step(action)

            if result.next_event is None:
                next_context = np.zeros_like(context)
                next_mask = np.zeros_like(mask)
                next_graph_state = feature_builder.build(result.state, env.current_time)
                done = True
            else:
                next_context, next_mask = env.observe(result.next_event)
                next_graph_state = feature_builder.build(result.state, result.next_event.event_time)

            episode_trajectory.append(
                {
                    "state": graph_state.copy(),
                    "context": context.copy(),
                    "action_mask": mask.copy(),
                    "action": action,
                    "log_prob": log_prob,
                    "reward": result.reward,
                    "next_state": next_graph_state.copy(),
                    "next_context": next_context.copy(),
                    "done": done,
                    "value": value,
                }
            )
            total_reward += result.reward
            total_reward_metro += result.reward_metro
            total_reward_bus += result.reward_bus
            total_reward_load += result.reward_load
            total_passengers += result.evacuated_passengers
            total_additional_bus_waiting += result.additional_bus_waiting
            if result.load_factor > 0:
                load_factors.append(result.load_factor)

            state_vec = result.state
            graph_state = next_graph_state
            context = next_context
            mask = next_mask

        rewards_history.append(total_reward)
        reward_metro_history.append(total_reward_metro)
        reward_bus_history.append(total_reward_bus)
        reward_load_history.append(total_reward_load)
        passenger_history.append(total_passengers)
        additional_bus_waiting_history.append(total_additional_bus_waiting)
        load_factor_history.append(float(np.mean(load_factors)) if load_factors else 0.0)
        trajectory_buffer.extend(episode_trajectory)

        if (episode + 1) % update_interval == 0 and trajectory_buffer:
            stats = agent.learn(trajectory_buffer)
            actor_loss_history.append(stats["actor_loss"])
            critic_loss_history.append(stats["critic_loss"])
            entropy_history.append(stats["entropy"])
            clip_frac_history.append(stats["clip_frac"])
            trajectory_buffer = []

    reporting_config = config.get("reporting", {})
    reported_rewards = [transform_metric(reporting_config, "reward", reward) for reward in rewards_history]
    reported_passengers = [
        int(transform_metric(reporting_config, "evacuated_passengers", value)) for value in passenger_history
    ]
    reported_bus_waiting = [
        int(transform_metric(reporting_config, "additional_bus_waiting", value))
        for value in additional_bus_waiting_history
    ]

    rewards_df = pd.DataFrame(
        {
            "Episode": np.arange(len(rewards_history)),
            "Raw_Reward": reported_rewards,
            "Smoothed_Reward": exponential_moving_average(reported_rewards),
            "Total_Passengers": reported_passengers,
            "Additional_Bus_Waiting": reported_bus_waiting,
            "Mean_Load_Factor": load_factor_history,
        }
    )
    rewards_df.to_csv(run_dir / "rewards_history.csv", index=False)

    loss_df = pd.DataFrame(
        {
            "Episode": np.arange(len(actor_loss_history)) * update_interval,
            "Actor_Loss": actor_loss_history,
            "Critic_Loss": critic_loss_history,
            "Entropy": entropy_history,
            "Clip_Fraction": clip_frac_history,
        }
    )
    loss_df.to_csv(run_dir / "loss_history.csv", index=False)
    torch.save(agent.checkpoint_state(), run_dir / "checkpoint.pt")
    plot_reward_loss_curves(rewards_df, loss_df, run_dir / "reward_loss_curves.png")

    print(f"Saved run: {run_dir}")
    print(f"Best reward: {rewards_df['Raw_Reward'].max():.3f}")
    print(f"Checkpoint: {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
