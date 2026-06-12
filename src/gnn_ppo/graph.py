from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


NODE_TYPES = ("metro_station", "bus_stop", "depot_dummy")
NODE_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(NODE_TYPES)}
EDGE_TYPES = ("metro_adjacency", "bus_network", "metro_bus_access", "depot_dispatch")
REQUIRED_NODE_COLUMNS = {"node_id", "node_type"}
REQUIRED_EDGE_COLUMNS = {"source_id", "target_id", "edge_type", "distance"}
REQUIRED_QUEUE_COLUMNS = {"time", "node_id", "queue"}


class GraphSchemaError(ValueError):
    pass


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _resolve_static_path(static_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else static_dir / path


def _require_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise GraphSchemaError(f"{path} is missing required columns: {sorted(missing)}")


def _normalise_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%S")
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    text = str(value)
    if len(text) == 8 and text[2] == ":" and text[5] == ":":
        return text
    parsed = pd.to_datetime(text)
    return parsed.strftime("%H:%M:%S")


def _paper_graph_missing_message(nodes_path: Path, edges_path: Path) -> str:
    return (
        "Paper multimodal graph inputs are required for the GNN in Section 5.1. "
        f"Missing {nodes_path} and/or {edges_path}. Add these files using the schema "
        "documented in data/static/PAPER_GRAPH_SCHEMA.md. Use graph.allow_legacy_fallback=true "
        "only for non-paper smoke tests."
    )


def load_paper_multimodal_graph(
    nodes_path: Path,
    edges_path: Path,
    *,
    distance_epsilon: float = 1e-6,
    make_undirected: bool = False,
) -> dict[str, Any]:
    nodes_df = _read_table(nodes_path)
    edges_df = _read_table(edges_path)
    _require_columns(nodes_df, REQUIRED_NODE_COLUMNS, nodes_path)
    _require_columns(edges_df, REQUIRED_EDGE_COLUMNS, edges_path)

    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    nodes_df["node_id"] = nodes_df["node_id"].astype(str)
    nodes_df["node_type"] = nodes_df["node_type"].astype(str)
    edges_df["source_id"] = edges_df["source_id"].astype(str)
    edges_df["target_id"] = edges_df["target_id"].astype(str)
    edges_df["edge_type"] = edges_df["edge_type"].astype(str)

    unknown_node_types = sorted(set(nodes_df["node_type"]) - set(NODE_TYPES))
    if unknown_node_types:
        raise GraphSchemaError(
            f"{nodes_path} has unsupported node_type values: {unknown_node_types}. "
            f"Allowed values are {list(NODE_TYPES)}."
        )
    unknown_edge_types = sorted(set(edges_df["edge_type"]) - set(EDGE_TYPES))
    if unknown_edge_types:
        raise GraphSchemaError(
            f"{edges_path} has unsupported edge_type values: {unknown_edge_types}. "
            f"Allowed values are {list(EDGE_TYPES)}."
        )

    node_ids = nodes_df["node_id"].tolist()
    duplicated_nodes = sorted(nodes_df.loc[nodes_df["node_id"].duplicated(), "node_id"].unique())
    if duplicated_nodes:
        raise GraphSchemaError(f"{nodes_path} has duplicated node_id values: {duplicated_nodes}")

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    unknown_sources = sorted(set(edges_df["source_id"]) - set(node_to_idx))
    unknown_targets = sorted(set(edges_df["target_id"]) - set(node_to_idx))
    if unknown_sources or unknown_targets:
        raise GraphSchemaError(
            f"{edges_path} references unknown nodes. "
            f"unknown source_id={unknown_sources}, unknown target_id={unknown_targets}"
        )

    distances = pd.to_numeric(edges_df["distance"], errors="coerce")
    if distances.isna().any() or (distances < 0).any():
        raise GraphSchemaError(f"{edges_path} distance must be numeric and non-negative.")

    num_nodes = len(nodes_df)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for row, distance in zip(edges_df.itertuples(index=False), distances):
        src = node_to_idx[str(row.source_id)]
        dst = node_to_idx[str(row.target_id)]
        weight = 1.0 / (float(distance) + float(distance_epsilon))
        adjacency[src, dst] = max(float(adjacency[src, dst]), weight)
        if make_undirected:
            adjacency[dst, src] = max(float(adjacency[dst, src]), weight)

    adj_hat = adjacency + torch.eye(num_nodes, dtype=torch.float32)
    degree = adj_hat.sum(dim=1).clamp_min(1e-12)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    normalized_adjacency = degree_inv_sqrt[:, None] * adj_hat * degree_inv_sqrt[None, :]

    node_type_indices = np.array([NODE_TYPE_TO_INDEX[t] for t in nodes_df["node_type"]], dtype=np.int64)
    node_type_one_hot = np.eye(len(NODE_TYPES), dtype=np.float32)[node_type_indices]
    station_to_node_idx: dict[int, int] = {}
    if "station_id" in nodes_df.columns:
        metro_rows = nodes_df[nodes_df["node_type"] == "metro_station"]
        for idx, row in metro_rows.iterrows():
            if pd.notna(row["station_id"]):
                station_to_node_idx[int(row["station_id"])] = int(idx)
    else:
        for idx, row in nodes_df[nodes_df["node_type"] == "metro_station"].iterrows():
            node_id = str(row["node_id"])
            if node_id.isdigit():
                station_to_node_idx[int(node_id)] = int(idx)

    return {
        "mode": "paper_multimodal",
        "num_nodes": num_nodes,
        "node_feature_dim": len(NODE_TYPES) + 1,
        "node_ids": node_ids,
        "node_to_idx": node_to_idx,
        "node_types": nodes_df["node_type"].tolist(),
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "adjacency": adjacency,
        "normalized_adjacency": normalized_adjacency,
        "node_type_one_hot": node_type_one_hot,
        "station_to_node_idx": station_to_node_idx,
    }


@dataclass
class PaperGraphFeatureBuilder:
    graph: dict[str, Any]
    station_features_df: pd.DataFrame
    regular_bus_queue_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        station_ids = pd.to_numeric(self.station_features_df["start_station"], errors="raise").to_numpy(dtype=int)
        self.edge_start_stations = station_ids
        if self.regular_bus_queue_df is not None:
            _require_columns(self.regular_bus_queue_df, REQUIRED_QUEUE_COLUMNS, Path("regular_bus_node_queues"))
            self.regular_bus_queue_df = self.regular_bus_queue_df.copy()
            self.regular_bus_queue_df["time"] = self.regular_bus_queue_df["time"].map(_normalise_time)
            self.regular_bus_queue_df["node_id"] = self.regular_bus_queue_df["node_id"].astype(str)

    def build(self, metro_edge_queues: np.ndarray, event_time: datetime | None) -> np.ndarray:
        features = np.zeros(
            (int(self.graph["num_nodes"]), int(self.graph["node_feature_dim"])),
            dtype=np.float32,
        )
        features[:, : len(NODE_TYPES)] = self.graph["node_type_one_hot"]
        queues = np.zeros(int(self.graph["num_nodes"]), dtype=np.float32)

        if self.regular_bus_queue_df is not None and event_time is not None:
            time_key = _normalise_time(event_time)
            rows = self.regular_bus_queue_df[self.regular_bus_queue_df["time"] == time_key]
            for row in rows.itertuples(index=False):
                node_id = str(row.node_id)
                if node_id not in self.graph["node_to_idx"]:
                    raise GraphSchemaError(f"regular_bus_node_queues references unknown node_id={node_id!r}.")
                node_idx = self.graph["node_to_idx"][node_id]
                queues[node_idx] = float(row.queue)

        station_to_node_idx: dict[int, int] = self.graph["station_to_node_idx"]
        if not station_to_node_idx:
            raise GraphSchemaError(
                "Paper graph nodes must provide metro station mapping. Add station_id for "
                "node_type=metro_station rows, or use numeric node_id values matching start_station."
            )

        if len(metro_edge_queues) != len(self.edge_start_stations):
            raise GraphSchemaError(
                "Metro queue vector length does not match station_features rows: "
                f"{len(metro_edge_queues)} != {len(self.edge_start_stations)}"
            )
        for station_id, queue in zip(self.edge_start_stations, metro_edge_queues):
            if int(station_id) not in station_to_node_idx:
                raise GraphSchemaError(
                    f"Metro station {int(station_id)} from station_features.xlsx is missing from paper graph nodes."
                )
            queues[station_to_node_idx[int(station_id)]] += float(queue)

        features[:, -1] = queues
        return features


def load_regular_bus_node_queues(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = _read_table(path)
    _require_columns(df, REQUIRED_QUEUE_COLUMNS, path)
    return df


def build_paper_graph_from_config(
    config: dict[str, Any],
    *,
    static_dir: Path,
    station_features_df: pd.DataFrame,
) -> tuple[dict[str, Any], PaperGraphFeatureBuilder]:
    graph_cfg = config.get("graph", {})
    nodes_path = _resolve_static_path(static_dir, graph_cfg.get("nodes", "multimodal_nodes.csv"))
    edges_path = _resolve_static_path(static_dir, graph_cfg.get("edges", "multimodal_edges.csv"))
    assert nodes_path is not None and edges_path is not None

    if not nodes_path.exists() or not edges_path.exists():
        if bool(graph_cfg.get("allow_legacy_fallback", False)):
            graph, _ = build_bipartite_graph_from_excel(static_dir / config["data"]["station_features"])
            builder = LegacyGraphFeatureBuilder(graph)
            return graph, builder
        raise GraphSchemaError(_paper_graph_missing_message(nodes_path, edges_path))

    graph = load_paper_multimodal_graph(
        nodes_path,
        edges_path,
        distance_epsilon=float(graph_cfg.get("distance_epsilon", 1e-6)),
        make_undirected=bool(graph_cfg.get("make_undirected", False)),
    )
    queue_path = _resolve_static_path(static_dir, graph_cfg.get("regular_bus_node_queues"))
    queue_df = load_regular_bus_node_queues(queue_path)
    builder = PaperGraphFeatureBuilder(graph, station_features_df, queue_df)
    return graph, builder


@dataclass
class LegacyGraphFeatureBuilder:
    graph: dict[str, Any]

    def build(self, metro_edge_queues: np.ndarray, event_time: datetime | None) -> np.ndarray:
        return metro_edge_queues.astype(np.float32)


def build_bipartite_graph_from_excel(excel_path: Path) -> tuple[dict[str, Any], int]:
    df = pd.read_excel(excel_path, usecols=[0, 1])
    df.columns = ["start_station", "end_line"]
    starts = df["start_station"].astype(int).tolist()
    ends = df["end_line"].astype(int).tolist()

    unique_starts = sorted(set(starts))
    unique_ends = sorted(set(ends))
    start_to_idx = {station: idx for idx, station in enumerate(unique_starts)}
    end_to_idx = {line: idx for idx, line in enumerate(unique_ends)}

    src_idx = torch.tensor([start_to_idx[s] for s in starts], dtype=torch.long)
    dst_idx = torch.tensor([end_to_idx[e] for e in ends], dtype=torch.long)
    graph = {
        "mode": "legacy_bipartite",
        "num_start": len(unique_starts),
        "num_end": len(unique_ends),
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "num_nodes_total": len(unique_starts) + len(unique_ends),
        "node_feature_dim": int(src_idx.shape[0]),
    }
    return graph, len(unique_starts)
