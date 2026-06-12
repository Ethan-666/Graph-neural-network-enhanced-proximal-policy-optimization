from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    allocate_od_to_station_edges,
    load_departure_info,
    load_line_features,
    load_station_features,
    od_file_for_time,
)


NO_DISPATCH = "no_dispatch"
RETURN = "return"
BRIDGE = "bridge"
STATUS_REGULAR = "regular"
STATUS_DEPOT = "depot"
STATUS_OFF_ROUTE = "off_route"


def parse_event_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, time):
        return datetime.strptime(value.strftime("%H:%M:%S"), "%H:%M:%S")
    if isinstance(value, str):
        return datetime.strptime(value, "%H:%M:%S")
    raise TypeError(f"Unsupported event time type: {type(value)!r}")


@dataclass(frozen=True)
class Action:
    action_id: int
    action_type: str
    edge_index: int | None = None
    start_station: int | None = None
    end_line: int | None = None
    end_station: int | None = None
    bus_line: int | None = None
    travel_time_min: float | None = None


@dataclass
class VehicleState:
    vehicle_id: str
    source_line: Any
    source_type: str
    status: str
    current_station: int | None
    original_line: Any
    recovery_target: Any
    capacity: int
    onboard: int = 0
    last_action_id: int | None = None


@dataclass
class Event:
    event_time: datetime
    vehicle_id: str
    event_type: str


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    reward_metro: float
    reward_bus: float
    reward_load: float
    evacuated_passengers: int
    additional_bus_waiting: int
    load_factor: float
    done: bool
    next_event: Event | None
    action: Action
    vehicle: VehicleState


class ActionCatalog:
    def __init__(self, station_features_df: pd.DataFrame) -> None:
        self.actions: list[Action] = [
            Action(0, NO_DISPATCH),
            Action(1, RETURN),
        ]
        for edge_index, row in station_features_df.reset_index(drop=True).iterrows():
            self.actions.append(
                Action(
                    action_id=len(self.actions),
                    action_type=BRIDGE,
                    edge_index=int(edge_index),
                    start_station=int(row["start_station"]),
                    end_line=int(row["end_line"]),
                    end_station=int(row["end_station"]),
                    bus_line=int(row["line"]),
                    travel_time_min=float(row["time"]),
                )
            )
        self.no_dispatch_id = 0
        self.return_id = 1
        self.bridge_offset = 2

    @property
    def size(self) -> int:
        return len(self.actions)

    def __getitem__(self, action_id: int) -> Action:
        return self.actions[int(action_id)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([action.__dict__ for action in self.actions])


class PaperAlignedBridgeEnv:
    def __init__(
        self,
        *,
        date: str,
        static_dir: Path,
        od_dir: Path,
        station_features_name: str,
        line_features_name: str,
        departure_info_name: str,
        env_config: dict[str, Any],
    ) -> None:
        self.date = date
        self.static_dir = static_dir
        self.od_dir = od_dir
        self.station_features_path = static_dir / station_features_name
        self.line_features_path = static_dir / line_features_name
        self.departure_info_path = static_dir / departure_info_name

        self.capacity = int(env_config["capacity"])
        self.omega_m = float(env_config.get("omega_m", 1.0))
        self.omega_b = float(env_config.get("omega_b", 1.0))
        self.omega_s = float(env_config.get("omega_s", 1200.0))
        self.bus_waiting_scale = float(env_config.get("bus_waiting_scale", 0.08))
        self.rho = float(env_config.get("demand_coverage_quantile", 0.5))
        self.max_dispatch_time_min = env_config.get("max_dispatch_time_min")
        self.max_dispatch_time_min = (
            None if self.max_dispatch_time_min is None else float(self.max_dispatch_time_min)
        )
        self.return_time_min = float(env_config.get("return_time_min", 8))
        self.depot_config = list(env_config.get("depot_config", []))
        self.use_depot = bool(env_config.get("use_depot", False))

        self.station_features_df = load_station_features(self.station_features_path).reset_index(drop=True)
        self.departure_info_array = load_departure_info(self.departure_info_path)
        self.busy_line, self.free_line_dict = self._load_line_spaces()
        self.initial_state = self._build_initial_state()
        self.action_catalog = ActionCatalog(self.station_features_df)

        self.state = self.initial_state.copy()
        self.vehicles: dict[str, VehicleState] = {}
        self.event_heap: list[tuple[datetime, int, Event]] = []
        self._event_seq = 0
        self.current_event: Event | None = None
        self.current_time: datetime | None = None
        self.end_time: datetime | None = None
        self.last_passenger_update_time: datetime | None = None

    def _load_line_spaces(self) -> tuple[np.ndarray, dict[Any, dict[str, Any]]]:
        line_features = load_line_features(self.line_features_path)
        busy_line = line_features.dropna(subset=["busy_line"])["busy_line"].to_numpy()
        free_line_dict: dict[Any, dict[str, Any]] = {}

        for _, row in line_features.iterrows():
            if pd.isna(row.get("free_line")) or pd.isna(row.get("dispatch_to")):
                continue
            free_line_dict[row["free_line"]] = {
                "dispatch_to": [int(x) for x in str(row["dispatch_to"]).split(",")],
                "source_type": "bus_line",
            }

        if self.use_depot:
            for depot in self.depot_config:
                free_line_dict[depot["line"]] = {
                    "dispatch_to": [int(x) for x in depot["dispatch_to"]],
                    "source_type": "depot",
                    "depot_name": depot["name"],
                }
        return busy_line, free_line_dict

    def _build_initial_state(self) -> np.ndarray:
        passenger_col = "分配人数" if "分配人数" in self.station_features_df.columns else "passenger_count"
        if passenger_col in self.station_features_df.columns:
            passenger_count = (
                pd.to_numeric(self.station_features_df[passenger_col], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=int)
            )
        else:
            passenger_count = np.zeros(len(self.station_features_df), dtype=int)

        state = np.empty((4, len(self.station_features_df)), dtype=int)
        state[0] = passenger_count
        state[1] = pd.to_numeric(self.station_features_df["start_station"], errors="raise").to_numpy(dtype=int)
        state[2] = pd.to_numeric(self.station_features_df["line"], errors="raise").to_numpy(dtype=int)
        state[3] = np.zeros(len(self.station_features_df), dtype=int)
        return state

    def sample_episode_window(self, duration_minutes: int) -> tuple[datetime, datetime]:
        start_hour = random.randint(16, 19)
        start_minute = random.choice([0, 10, 20, 30, 40, 50])
        if start_hour == 19 and start_minute > 30:
            start_minute = 30
        start_time = datetime.strptime(f"{start_hour:02d}:{start_minute:02d}:00", "%H:%M:%S")
        return start_time, start_time + timedelta(minutes=duration_minutes)

    def reset(self, start_time: datetime, duration_minutes: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray, Event]:
        self.state = self.initial_state.copy()
        self.vehicles = {}
        self.event_heap = []
        self._event_seq = 0
        self.current_event = None
        self.current_time = start_time
        self.end_time = start_time + timedelta(minutes=duration_minutes)
        self.last_passenger_update_time = start_time - timedelta(minutes=10)

        self._seed_regular_vehicle_events(start_time, self.end_time)
        if self.use_depot:
            self._seed_depot_vehicle_events(start_time, self.end_time)

        event = self.next_event()
        if event is None:
            raise RuntimeError("Episode has no controllable events.")
        observation, mask = self.observe(event)
        return self.state[0].copy(), observation, mask, event

    def _seed_regular_vehicle_events(self, start_time: datetime, end_time: datetime) -> None:
        for idx, row in enumerate(self.departure_info_array):
            event_time = parse_event_time(row[1])
            if not (start_time <= event_time <= end_time):
                continue
            vehicle_id = f"regular_{idx}_{row[0]}_{event_time.strftime('%H%M%S')}"
            vehicle = VehicleState(
                vehicle_id=vehicle_id,
                source_line=row[0],
                source_type="bus_line",
                status=STATUS_REGULAR,
                current_station=None,
                original_line=row[0],
                recovery_target=row[0],
                capacity=self.capacity,
            )
            self.vehicles[vehicle_id] = vehicle
            self._push_event(Event(event_time, vehicle_id, "regular_available"))

    def _seed_depot_vehicle_events(self, start_time: datetime, end_time: datetime) -> None:
        for depot in self.depot_config:
            for idx in range(int(depot["n_buses"])):
                ready_time = start_time + timedelta(
                    minutes=int(depot["prep_min"]) + idx * int(depot["headway_min"])
                )
                if not (start_time <= ready_time <= end_time):
                    continue
                vehicle_id = f"depot_{depot['line']}_{idx + 1}_{ready_time.strftime('%H%M%S')}"
                vehicle = VehicleState(
                    vehicle_id=vehicle_id,
                    source_line=depot["line"],
                    source_type="depot",
                    status=STATUS_DEPOT,
                    current_station=None,
                    original_line=depot["line"],
                    recovery_target=depot["line"],
                    capacity=self.capacity,
                )
                self.vehicles[vehicle_id] = vehicle
                self._push_event(Event(ready_time, vehicle_id, "depot_available"))

    def _push_event(self, event: Event) -> None:
        if self.end_time is not None and event.event_time > self.end_time:
            return
        heapq.heappush(self.event_heap, (event.event_time, self._event_seq, event))
        self._event_seq += 1

    def next_event(self) -> Event | None:
        while self.event_heap:
            event_time, _seq, event = heapq.heappop(self.event_heap)
            if self.end_time is not None and event_time > self.end_time:
                return None
            self._update_passengers_until(event_time)
            self.current_time = event_time
            self.current_event = event
            return event
        return None

    def _update_passengers_until(self, target_time: datetime) -> None:
        if self.last_passenger_update_time is None:
            return
        next_update = self._next_ten_minute_time(self.last_passenger_update_time)
        max_update = datetime.strptime("20:00:00", "%H:%M:%S")
        while next_update <= target_time and next_update <= max_update:
            self._add_passenger_arrivals(next_update)
            self.last_passenger_update_time = next_update
            next_update = self._next_ten_minute_time(self.last_passenger_update_time)

    @staticmethod
    def _next_ten_minute_time(value: datetime) -> datetime:
        next_trigger_minute = (value.minute // 10 + 1) * 10
        return value.replace(minute=0, second=0) + timedelta(minutes=next_trigger_minute)

    def _add_passenger_arrivals(self, trigger_time: datetime) -> None:
        path = od_file_for_time(self.od_dir, self.date, trigger_time)
        if not path.exists():
            raise FileNotFoundError(f"Missing OD file for passenger update: {path}")
        od_df = pd.read_csv(path)
        passenger_count = allocate_od_to_station_edges(od_df, self.station_features_df)
        if self.state[0].shape != passenger_count.shape:
            raise ValueError(
                f"Shape mismatch: state[0].shape={self.state[0].shape}, passenger_count.shape={passenger_count.shape}"
            )
        self.state[0] += passenger_count

    def observe(self, event: Event) -> tuple[np.ndarray, np.ndarray]:
        vehicle = self.vehicles[event.vehicle_id]
        context = self.vehicle_context(vehicle, event.event_time)
        mask = self.action_mask(vehicle)
        return context, mask

    def vehicle_context(self, vehicle: VehicleState, event_time: datetime) -> np.ndarray:
        minutes = event_time.hour * 60 + event_time.minute
        line_value = self._line_to_float(vehicle.source_line)
        return np.asarray(
            [
                minutes / (24.0 * 60.0),
                1.0 if vehicle.status == STATUS_REGULAR else 0.0,
                1.0 if vehicle.status == STATUS_DEPOT else 0.0,
                1.0 if vehicle.status == STATUS_OFF_ROUTE else 0.0,
                1.0 if vehicle.source_type == "depot" else 0.0,
                min(line_value / 1000.0, 1.0),
                vehicle.capacity / max(float(self.capacity), 1.0),
                vehicle.onboard / max(float(vehicle.capacity), 1.0),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _line_to_float(line: Any) -> float:
        try:
            return float(line)
        except (TypeError, ValueError):
            return 999.0

    def action_mask(self, vehicle: VehicleState) -> np.ndarray:
        mask = np.zeros(self.action_catalog.size, dtype=bool)
        if vehicle.status in {STATUS_REGULAR, STATUS_DEPOT}:
            mask[self.action_catalog.no_dispatch_id] = True
        if vehicle.status == STATUS_OFF_ROUTE:
            mask[self.action_catalog.return_id] = True

        feasible_bridge_ids = self._feasible_bridge_action_ids(vehicle)
        filtered_bridge_ids = self._apply_demand_coverage_filter(feasible_bridge_ids)
        mask[filtered_bridge_ids] = True

        if not mask.any():
            mask[self.action_catalog.no_dispatch_id] = True
        return mask

    def _feasible_bridge_action_ids(self, vehicle: VehicleState) -> list[int]:
        allowed_rows = self._candidate_edge_indices_for_vehicle(vehicle)
        action_ids: list[int] = []
        for edge_index in allowed_rows:
            if self.state[0][edge_index] <= 0:
                continue
            action = self.action_catalog.actions[self.action_catalog.bridge_offset + edge_index]
            if self.max_dispatch_time_min is not None and action.travel_time_min is not None:
                if action.travel_time_min > self.max_dispatch_time_min:
                    continue
            action_ids.append(action.action_id)

        if not action_ids and self.max_dispatch_time_min is not None:
            for edge_index in allowed_rows:
                if self.state[0][edge_index] > 0:
                    action_ids.append(self.action_catalog.bridge_offset + int(edge_index))
        return action_ids

    def _candidate_edge_indices_for_vehicle(self, vehicle: VehicleState) -> np.ndarray:
        source_line = vehicle.source_line
        if source_line in self.busy_line.tolist():
            return np.where(self.state[2] == source_line)[0]

        if source_line in self.free_line_dict:
            dispatch_to = self.free_line_dict[source_line]["dispatch_to"]
            chunks = [np.where(self.state[1] == int(station))[0] for station in dispatch_to]
            return np.concatenate(chunks) if chunks else np.array([], dtype=int)

        if vehicle.status == STATUS_OFF_ROUTE:
            if vehicle.original_line in self.free_line_dict:
                dispatch_to = self.free_line_dict[vehicle.original_line]["dispatch_to"]
                chunks = [np.where(self.state[1] == int(station))[0] for station in dispatch_to]
                return np.concatenate(chunks) if chunks else np.array([], dtype=int)
            if vehicle.original_line in self.busy_line.tolist():
                return np.where(self.state[2] == vehicle.original_line)[0]

        return np.array([], dtype=int)

    def _apply_demand_coverage_filter(self, action_ids: list[int]) -> list[int]:
        if not action_ids:
            return []
        demands = np.asarray(
            [self.state[0][self.action_catalog[action_id].edge_index] for action_id in action_ids],
            dtype=float,
        )
        positive = demands > 0
        if not positive.any():
            return []

        action_ids_arr = np.asarray(action_ids)[positive]
        demands = demands[positive]
        order = np.argsort(demands)[::-1]
        sorted_action_ids = action_ids_arr[order]
        sorted_demands = demands[order]
        threshold = self.rho * sorted_demands.sum()
        cumulative = np.cumsum(sorted_demands)
        keep_count = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        return [int(x) for x in sorted_action_ids[:keep_count]]

    def step(self, action_id: int) -> StepResult:
        if self.current_event is None:
            raise RuntimeError("No current event. Call reset() or next_event() first.")

        event = self.current_event
        vehicle = self.vehicles[event.vehicle_id]
        action_time = self.current_time
        if action_time is None:
            raise RuntimeError("Current event time is not initialized.")
        mask = self.action_mask(vehicle)
        if action_id < 0 or action_id >= self.action_catalog.size or not mask[action_id]:
            raise ValueError(f"Action {action_id} is not feasible for vehicle {vehicle.vehicle_id}.")

        action = self.action_catalog[action_id]
        evacuated = 0
        additional_bus_waiting = 0
        load_factor = 0.0

        if action.action_type == NO_DISPATCH:
            vehicle.last_action_id = action.action_id
        elif action.action_type == RETURN:
            vehicle.status = STATUS_REGULAR if vehicle.source_type == "bus_line" else STATUS_DEPOT
            vehicle.current_station = None
            vehicle.onboard = 0
            vehicle.last_action_id = action.action_id
            return_time = self.current_time + timedelta(minutes=self.return_time_min)
            self._push_event(Event(return_time, vehicle.vehicle_id, "return_complete"))
        elif action.action_type == BRIDGE:
            additional_bus_waiting = self._estimate_additional_bus_waiting(vehicle, action)
            evacuated, load_factor = self._serve_bridge_action(vehicle, action)
            service_time = max(float(action.travel_time_min or 0.0), 1.0)
            vehicle.status = STATUS_OFF_ROUTE
            vehicle.current_station = action.end_station
            vehicle.onboard = 0
            vehicle.last_action_id = action.action_id
            if vehicle.source_type == "bus_line":
                self._push_event(
                    Event(
                        self.current_time + timedelta(minutes=service_time),
                        vehicle.vehicle_id,
                        "bridge_complete",
                    )
                )
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

        metro_queue_start = float(np.sum(self.state[0]))
        next_event = self.next_event()
        done = next_event is None
        if done and self.end_time is not None:
            self._update_passengers_until(self.end_time)
            reward_end_time = self.end_time
        else:
            reward_end_time = next_event.event_time if next_event is not None else action_time

        metro_queue_end = float(np.sum(self.state[0]))
        transition_minutes = max((reward_end_time - action_time).total_seconds() / 60.0, 0.0)
        reward_metro, reward_bus, reward_load, reward = self._paper_reward_components(
            metro_queue_start=metro_queue_start,
            metro_queue_end=metro_queue_end,
            additional_bus_waiting=additional_bus_waiting,
            load_factor=load_factor,
            transition_minutes=transition_minutes,
        )

        return StepResult(
            state=self.state[0].copy(),
            reward=float(reward),
            reward_metro=float(reward_metro),
            reward_bus=float(reward_bus),
            reward_load=float(reward_load),
            evacuated_passengers=int(evacuated),
            additional_bus_waiting=int(additional_bus_waiting),
            load_factor=float(load_factor),
            done=done,
            next_event=next_event,
            action=action,
            vehicle=vehicle,
        )

    def _serve_bridge_action(self, vehicle: VehicleState, action: Action) -> tuple[int, float]:
        assert action.edge_index is not None
        edge_index = action.edge_index
        before = int(self.state[0][edge_index])
        evacuated = min(before, vehicle.capacity)
        self.state[0][edge_index] = max(0, before - evacuated)
        self.state[3][edge_index] += 1
        load_factor = evacuated / max(float(vehicle.capacity), 1.0)
        vehicle.onboard = evacuated
        return evacuated, load_factor

    def _paper_reward_components(
        self,
        *,
        metro_queue_start: float,
        metro_queue_end: float,
        additional_bus_waiting: int,
        load_factor: float,
        transition_minutes: float,
    ) -> tuple[float, float, float, float]:
        metro_waiting = 0.5 * (metro_queue_start + metro_queue_end) * transition_minutes
        bus_waiting = float(additional_bus_waiting) * transition_minutes
        reward_metro = -self.omega_m * metro_waiting
        reward_bus = -self.omega_b * bus_waiting
        reward_load = self.omega_s * load_factor
        return reward_metro, reward_bus, reward_load, reward_metro + reward_bus + reward_load

    def _estimate_additional_bus_waiting(self, vehicle: VehicleState, action: Action) -> int:
        if vehicle.source_type != "bus_line":
            return 0
        candidate_indices = self._candidate_edge_indices_for_vehicle(vehicle)
        if len(candidate_indices) == 0:
            return int(vehicle.capacity)
        if action.edge_index is not None and action.edge_index in set(candidate_indices.tolist()):
            pressure = int(self.state[0][action.edge_index])
        else:
            pressure = int(np.max(self.state[0][candidate_indices]))
        impacted = min(vehicle.capacity, max(pressure, 0)) * self.bus_waiting_scale
        return int(round(impacted))
