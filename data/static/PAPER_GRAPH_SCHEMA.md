# Paper Multimodal Graph Input Schema

The paper-aligned GNN uses the graph definition in Section 5.1:

`G = (N, E_G, D_G)`, where nodes are metro stations, bus stops, and depot dummy stops; edges are metro-adjacency, bus-network, metro-bus access, and depot-dispatch links; each edge stores distance `d_ij`; the encoder uses `A_ij = 1 / (d_ij + epsilon)` and `D_hat^-1/2 A_hat D_hat^-1/2`.

Place these files in `data/static/` or change `configs/default.json -> graph`.

## `multimodal_nodes.csv`

Required columns:

| column | type | description |
| --- | --- | --- |
| `node_id` | string | Unique node identifier used by edges and queue files. |
| `node_type` | string | One of `metro_station`, `bus_stop`, `depot_dummy`. |

Required for metro station rows:

| column | type | description |
| --- | --- | --- |
| `station_id` | integer | Must match `station_features.xlsx:start_station`. This maps the simulator OD queues to graph nodes. |

Recommended optional columns:

| column | type | description |
| --- | --- | --- |
| `name` | string | Station, stop, or depot name. |
| `stop_id` | string | Original bus-stop identifier for `bus_stop` nodes. |
| `route_id` | string | Bus route or line identifier if the stop is route-specific. |
| `depot_id` | string | Depot identifier for `depot_dummy` nodes. |
| `lon`, `lat` | float | Coordinates used to construct or audit access links. |

Example:

```csv
node_id,node_type,station_id,stop_id,route_id,depot_id,name,lon,lat
M7,metro_station,7,,,,Station 7,116.20,39.90
B115_03,bus_stop,,115_03,115,,Route 115 stop 3,116.21,39.90
D_LAOSHAN,depot_dummy,,,,DEPOT_LAOSHAN,Laoshan bus depot,116.18,39.91
```

## `multimodal_edges.csv`

Required columns:

| column | type | description |
| --- | --- | --- |
| `source_id` | string | Source node id from `multimodal_nodes.csv`. |
| `target_id` | string | Target node id from `multimodal_nodes.csv`. |
| `edge_type` | string | One of `metro_adjacency`, `bus_network`, `metro_bus_access`, `depot_dispatch`. |
| `distance` | float | Non-negative edge distance or dispatch travel-time distance used in `A_ij = 1/(d_ij+epsilon)`. |

The loader treats edges as directed. If an edge is bidirectional in the paper graph, include both directions or set `graph.make_undirected=true` for a non-directed import.

Example:

```csv
source_id,target_id,edge_type,distance
M7,M8,metro_adjacency,1.20
M8,M7,metro_adjacency,1.20
B115_03,B115_04,bus_network,0.65
M7,B115_03,metro_bus_access,0.22
D_LAOSHAN,B115_03,depot_dispatch,3.00
```

## `regular_bus_node_queues.csv` Optional

Required columns:

| column | type | description |
| --- | --- | --- |
| `time` | string | `HH:MM:SS`, aligned to the simulator grid. |
| `node_id` | string | A `bus_stop` node id from `multimodal_nodes.csv`. |
| `queue` | float | Regular-bus waiting queue `q^B_{r,b}(t)` at that stop and time. |

Example:

```csv
time,node_id,queue
09:00:00,B115_03,18
09:10:00,B115_03,21
```

Metro station queue features are built from the environment's current disrupted OD queues and aggregated to `metro_station` nodes through `station_id`. For fully paper-consistent regular-bus node features, provide regular-bus stop queues from the regular-bus simulator or reconstructed smart-card demand process.
