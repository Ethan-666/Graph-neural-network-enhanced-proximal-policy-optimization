from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import data_paths, load_config
from .data import validate_prepared_dataset
from .env import PaperAlignedBridgeEnv
from .graph import build_paper_graph_from_config
from .models import PPOAgent
from .reporting import transform_metric
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained masked GNN-PPO dispatcher.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-depot", action="store_true")
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


def evaluate_window(
    env: PaperAlignedBridgeEnv,
    agent: PPOAgent,
    feature_builder: Any,
    start_time: datetime,
    duration: int,
    reporting_config: dict[str, Any],
) -> dict[str, Any]:
    state_vec, context, mask, _event = env.reset(start_time, duration)
    graph_state = feature_builder.build(state_vec, env.current_time)
    total_reward = 0.0
    total_passengers = 0
    total_additional_bus_waiting = 0
    depot_used = 0
    load_factors: list[float] = []
    action_rows: list[dict[str, Any]] = []
    done = False

    while not done:
        action_time = env.current_time
        action_id = agent.greedy_action(graph_state, context, mask)
        result = env.step(action_id)
        if result.action.action_type == "bridge" and result.vehicle.source_type == "depot":
            depot_used += 1
        if result.load_factor > 0:
            load_factors.append(result.load_factor)
        total_additional_bus_waiting += result.additional_bus_waiting
        reported_reward = transform_metric(reporting_config, "reward", result.reward)
        reported_evacuated = int(
            transform_metric(reporting_config, "evacuated_passengers", result.evacuated_passengers)
        )
        reported_bus_waiting = int(
            transform_metric(reporting_config, "additional_bus_waiting", result.additional_bus_waiting)
        )
        action_rows.append(
            {
                "time": action_time.strftime("%H:%M:%S") if action_time else "",
                "vehicle_id": result.vehicle.vehicle_id,
                "source_type": result.vehicle.source_type,
                "action_id": result.action.action_id,
                "action_type": result.action.action_type,
                "edge_index": result.action.edge_index,
                "start_station": result.action.start_station,
                "end_line": result.action.end_line,
                "end_station": result.action.end_station,
                "bus_line": result.action.bus_line,
                "evacuated_passengers": reported_evacuated,
                "additional_bus_waiting": reported_bus_waiting,
                "load_factor": result.load_factor,
                "reward": reported_reward,
            }
        )
        total_reward += result.reward
        total_passengers += result.evacuated_passengers
        if result.next_event is None:
            done = True
        else:
            context, mask = env.observe(result.next_event)
            state_vec = result.state
            graph_state = feature_builder.build(state_vec, result.next_event.event_time)

    reported_total_reward = transform_metric(reporting_config, "reward", total_reward)
    reported_total_passengers = int(
        transform_metric(reporting_config, "evacuated_passengers", total_passengers)
    )
    reported_total_bus_waiting = int(
        transform_metric(reporting_config, "additional_bus_waiting", total_additional_bus_waiting)
    )
    return {
        "start_time": start_time.strftime("%H:%M:%S"),
        "reward": reported_total_reward,
        "evacuated_passengers": reported_total_passengers,
        "additional_bus_waiting": reported_total_bus_waiting,
        "mean_load_factor": float(np.mean(load_factors)) if load_factors else 0.0,
        "depot_used": depot_used,
        "actions": action_rows,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.no_depot:
        config["env"]["use_depot"] = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = build_env(config)
    paths = data_paths(config)
    graph, feature_builder = build_paper_graph_from_config(
        config,
        static_dir=paths["static_dir"],
        station_features_df=env.station_features_df,
    )
    agent = PPOAgent(
        graph,
        config["model"],
        device,
        action_dim=env.action_catalog.size,
        context_dim=8,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.load_checkpoint_state(checkpoint)

    output_dir = ensure_dir(args.output_dir or args.checkpoint.parent / "evaluation")
    (output_dir / "config_used_evaluation.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    duration = int(config["data"]["episode_duration_minutes"])
    reporting_config = config.get("reporting", {})
    summaries = []
    for window in config["data"]["evaluation_windows"]:
        start_time = datetime.strptime(window, "%H:%M:%S")
        result = evaluate_window(env, agent, feature_builder, start_time, duration, reporting_config)
        actions = result.pop("actions")
        summaries.append(result)
        pd.DataFrame(actions).to_csv(output_dir / f"actions_{window.replace(':', '')}.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved evaluation: {output_dir}")


if __name__ == "__main__":
    main()
