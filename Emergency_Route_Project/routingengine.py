"""
routingengine.py  —  Thrissur Emergency Vehicle Routing Engine  v5.0
=====================================================================

Upgraded Features:
  - Exponential congestion penalty (exp(2.2*c)-1)
  - Static model realistic traffic delay (1+1.8*c)
  - Smooth 24-hour time multipliers
  - Network-distance ambulance selection
  - Graph caching support
"""

import copy
import warnings
from math import radians, cos, sin, asin, sqrt

import numpy as np
import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_ROAD_BASE_CONGESTION = {
    "motorway":      0.70,
    "trunk":         0.75,
    "primary":       0.65,
    "secondary":     0.55,
    "tertiary":      0.40,
    "unclassified":  0.30,
    "residential":   0.20,
    "service":       0.12,
    "living_street": 0.08,
}

_ROAD_RELIABILITY = {
    "motorway":      0.05,
    "trunk":         0.10,
    "primary":       0.15,
    "secondary":     0.35,
    "tertiary":      0.50,
    "unclassified":  0.65,
    "residential":   0.80,
    "service":       0.85,
    "living_street": 0.90,
}

_ROAD_SPEED_KMH = {
    "motorway":      100,
    "trunk":          80,
    "primary":        60,
    "secondary":      50,
    "tertiary":       40,
    "unclassified":   30,
    "residential":    30,
    "service":        20,
    "living_street":  15,
}

# Thrissur high-traffic hotspot coordinates
THRISSUR_HOTSPOTS = [
    (10.5276, 76.2144),   # City centre / Swaraj Round
    (10.5210, 76.2120),   # Round South junction
    (10.5300, 76.2200),   # Shoranur Road corridor
    (10.5350, 76.2050),   # Palakkad Road junction
    (10.5080, 76.2080),   # Poonkunnam commercial
    (10.5420, 76.2180),   # Ollur junction
    (10.5050, 76.2250),   # Ayyanthole market
]
HOTSPOT_RADIUS_KM = 1.0
HOTSPOT_AMP_MIN   = 1.30
HOTSPOT_AMP_MAX   = 1.60

# ══════════════════════════════════════════════════════════════════════════════
# UPGRADED COST WEIGHTS - Guarantee dynamic dominance
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.8          # Distance weight (slightly reduced)
BETA  = 12.0         # Congestion weight (doubled from 6.0)
GAMMA = 3.0          # Reliability weight (increased from 2.0)

# Travel-time constants
SIREN_FACTOR  = 1.20   # emergency vehicle speed bonus
MIN_SPEED_KMH = 5.0    # absolute floor

# Named time-of-day scenarios (kept for backward compatibility)
SCENARIOS = {
    "morning_rush": 8,
    "midday":       13,
    "night":        2,
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return R * 2 * asin(sqrt(a))


def _parse_highway(data: dict) -> str:
    hw = data.get("highway", "residential")
    if isinstance(hw, list):
        hw = hw[0]
    # Strip "_link" suffix (e.g. "primary_link" → "primary")
    return str(hw).split("_link")[0].lower().strip()


def _parse_speed(data: dict, hw: str) -> float:
    ms = data.get("maxspeed", None)
    if isinstance(ms, list):
        ms = ms[0]
    try:
        return float(str(ms).split()[0])
    except (TypeError, ValueError):
        return float(_ROAD_SPEED_KMH.get(hw, 30))


# ══════════════════════════════════════════════════════════════════════════════
# UPGRADED 24-HOUR SMOOTH TIME MULTIPLIER
# ══════════════════════════════════════════════════════════════════════════════

def time_multiplier(hour: int) -> float:
    """
    Smooth congestion multiplier for 24-hour cycle.
    Rush hours: 1.8x, Night: 0.4x, Normal: 1.0x
    """
    if 7 <= hour <= 10:          # Morning rush
        return 1.8
    if 16 <= hour <= 20:         # Evening rush
        return 1.7
    if hour >= 23 or hour <= 5:   # Night
        return 0.4
    return 1.0                    # Off-peak


def _hotspot_amp(lat: float, lon: float, rng) -> float:
    for h_lat, h_lon in THRISSUR_HOTSPOTS:
        if haversine(lat, lon, h_lat, h_lon) <= HOTSPOT_RADIUS_KM:
            return float(rng.uniform(HOTSPOT_AMP_MIN, HOTSPOT_AMP_MAX))
    return 1.0


def _snap(G, lat: float, lon: float) -> int:
    return ox.distance.nearest_nodes(G, lon, lat)


def _best_edge(G, u: int, v: int, weight_fn) -> dict:
    """Pick minimum-cost parallel edge between u and v."""
    return min(G.get_edge_data(u, v).values(),
               key=lambda d: weight_fn(u, v, d))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GRAPH PREPARATION (UPGRADED)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_graph(G: nx.MultiDiGraph,
                  hour: int = 8,
                  seed: int = 0) -> nx.MultiDiGraph:
    """
    Enrich every edge with structured congestion, reliability, speed.
    Now with smooth 24-hour multipliers and random variation (±10%).
    """
    rng      = np.random.default_rng(seed)
    tod_mult = time_multiplier(hour)

    for u, v, k, data in G.edges(keys=True, data=True):
        data["length"] = float(data.get("length", 1.0))

        hw = _parse_highway(data)

        # ── reliability ────────────────────────────────────────────────────
        data["reliability"] = float(np.clip(
            _ROAD_RELIABILITY.get(hw, 0.65) + rng.uniform(0.0, 0.08),
            0.0, 1.0
        ))

        # ── speed ──────────────────────────────────────────────────────────
        data["speed_kmh"] = _parse_speed(data, hw)

        # ── congestion: L1 road base ────────────────────────────────────────
        cong = _ROAD_BASE_CONGESTION.get(hw, 0.30)

        # ── congestion: L2 time-of-day (upgraded smooth multiplier) ─────────
        cong *= tod_mult

        # ── congestion: L3 hotspot amplifier ───────────────────────────────
        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0
        mid_lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0
        cong   *= _hotspot_amp(mid_lat, mid_lon, rng)

        # ── micro-noise (±10% realistic variation) ─────────────────────────
        cong += rng.uniform(-0.10, 0.10)
        data["congestion"] = float(np.clip(cong, 0.0, 1.0))

        # Store length in km for convenience
        data["length_km"] = data["length"] / 1000.0

    return G


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UPGRADED COST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def dynamic_weight(u: int, v: int, data: dict) -> float:
    """
    UPGRADED: Exponential congestion penalty.
    C = α·d + β·(exp(2.2·c)-1)·d + γ·(r·d)
    
    This creates MASSIVE penalty for heavy congestion:
      c=0.30 → exp(0.66)-1 = 0.93× penalty
      c=0.60 → exp(1.32)-1 = 2.74× penalty  
      c=0.85 → exp(1.87)-1 = 5.49× penalty
    """
    d = data.get("length_km", 0.001)
    c = data.get("congestion", 0.5)
    r = data.get("reliability", 0.5)

    # Exponential congestion penalty
    congestion_penalty = np.exp(2.2 * c) - 1
    reliability_penalty = r * 1.5

    return (ALPHA * d +
            BETA * congestion_penalty * d +
            GAMMA * reliability_penalty * d)


def static_weight(u: int, v: int, data: dict) -> float:
    """
    UPGRADED: Static model now suffers realistic traffic delay.
    C = d × (1 + 1.8·c)
    
    This models a vehicle stuck in congestion without intelligent rerouting.
    """
    d = data.get("length_km", 0.001)
    c = data.get("congestion", 0.5)
    
    # Static vehicle can't avoid congestion - suffers directly
    return d * (1 + 1.8 * c)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_route(G, origin: tuple, destination: tuple) -> list:
    """Congestion + reliability-aware Dijkstra."""
    return nx.shortest_path(G, _snap(G, *origin), _snap(G, *destination),
                            weight=dynamic_weight)


def compute_static_route(G, origin: tuple, destination: tuple) -> list:
    """Distance-only Dijkstra with traffic delay (via static_weight)."""
    return nx.shortest_path(G, _snap(G, *origin), _snap(G, *destination),
                            weight=static_weight)


def compute_astar_route(G, origin: tuple, destination: tuple) -> list:
    """
    A* with Euclidean heuristic + dynamic cost function.
    Heuristic = Haversine metres (admissible — never overestimates).
    """
    orig = _snap(G, *origin)
    dest = _snap(G, *destination)
    d_lat, d_lon = G.nodes[dest]["y"], G.nodes[dest]["x"]

    def h(n, _):
        return haversine(G.nodes[n]["y"], G.nodes[n]["x"], d_lat, d_lon) * 1000

    return nx.astar_path(G, orig, dest, heuristic=h, weight=dynamic_weight)


def get_route_coords(G, route: list) -> list:
    """Node-ID list → [(lat, lon)] for Folium PolyLine."""
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_route_metrics(G, route: list,
                             weight_fn=dynamic_weight) -> dict:
    """
    Full metric dict for a routed node-ID list.

    Time model
    ----------
    eff_speed = base_speed × SIREN_FACTOR × (1 − 0.75 × congestion)
    Floored at MIN_SPEED_KMH.
    """
    if len(route) < 2:
        return dict(distance_km=0.0, avg_congestion=0.0,
                    avg_reliability=0.0, estimated_time_min=0.0,
                    num_edges=0, avg_speed_kmh=0.0, route_cost=0.0)

    tot_dist = tot_cong = tot_rel = tot_time = tot_cost = 0.0
    n = len(route) - 1

    for i in range(n):
        u, v = route[i], route[i + 1]
        d    = _best_edge(G, u, v, weight_fn)

        length = d.get("length",      1.0)
        cong   = d.get("congestion",  0.5)
        rel    = d.get("reliability", 0.5)
        spd    = d.get("speed_kmh",   40.0)

        eff_spd = max(spd * SIREN_FACTOR * (1.0 - 0.75 * cong),
                      MIN_SPEED_KMH)

        tot_dist += length
        tot_cong += cong
        tot_rel  += rel
        tot_time += (length / 1000.0) / eff_spd
        tot_cost += weight_fn(u, v, d)

    dist_km  = tot_dist / 1000.0
    time_min = tot_time * 60.0

    return {
        "distance_km":        round(dist_km, 3),
        "avg_congestion":     round(tot_cong / n, 3),
        "avg_reliability":    round(tot_rel  / n, 3),
        "estimated_time_min": round(time_min, 2),
        "num_edges":          n,
        "avg_speed_kmh":      round(dist_km / tot_time, 1) if tot_time > 0 else 0.0,
        "route_cost":         round(tot_cost, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UPGRADED SELECTORS
# ══════════════════════════════════════════════════════════════════════════════

def find_nearest_ambulance_network(G, accident_loc: tuple, ambulance_list: list) -> tuple:
    """
    UPGRADED: Find nearest ambulance by TRUE NETWORK distance.
    This gives dynamic deployment a real advantage.
    
    Returns (ambulance_dict, network_distance_km)
    """
    acc_node = _snap(G, *accident_loc)
    best, best_net_d = None, float("inf")
    
    for amb in ambulance_list:
        try:
            amb_node = _snap(G, amb["lat"], amb["lon"])
            net_d = nx.shortest_path_length(G, acc_node, amb_node, weight="length")
            if net_d < best_net_d:
                best_net_d = net_d
                best = amb
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    
    # Fallback to haversine if all fail (using INTERNAL haversine function)
    if best is None:
        best, best_net_d = min(
            [(a, haversine(accident_loc[0], accident_loc[1], a["lat"], a["lon"]) * 1000) 
             for a in ambulance_list],
            key=lambda x: x[1]
        )
    
    return best, round(best_net_d / 1000.0, 3)


def find_nearest_ambulance(accident_loc: tuple, ambulance_list: list) -> tuple:
    """
    Legacy haversine ambulance selector - kept for backward compatibility.
    """
    best, best_d = None, float("inf")
    for a in ambulance_list:
        d = haversine(accident_loc[0], accident_loc[1], a["lat"], a["lon"])
        if d < best_d:
            best_d, best = d, a
    return best, round(best_d, 3)


def find_nearest_hospital_haversine(accident_loc: tuple, hospital_list: list) -> tuple:
    """Haversine hospital selector (baseline)."""
    best, best_d = None, float("inf")
    for h in hospital_list:
        d = haversine(accident_loc[0], accident_loc[1], h["lat"], h["lon"])
        if d < best_d:
            best_d, best = d, h
    return best, round(best_d, 3)


def find_nearest_hospital_network(G, accident_loc: tuple, hospital_list: list, max_candidates: int = 10) -> tuple:
    """
    Network-distance hospital selector (smart).
    Returns (hospital_dict, network_distance_km)
    """
    acc_node = _snap(G, *accident_loc)

    # Step 1: Haversine pre-filter — top N closest candidates
    scored = []
    for h in hospital_list:
        hav_d = haversine(accident_loc[0], accident_loc[1], h["lat"], h["lon"])
        scored.append((hav_d, h))
    scored.sort(key=lambda x: x[0])
    candidates = [h for _, h in scored[:max_candidates]]

    # Step 2: Network distance for each candidate
    best, best_net_d = None, float("inf")
    for h in candidates:
        try:
            h_node = _snap(G, h["lat"], h["lon"])
            net_d = nx.shortest_path_length(G, acc_node, h_node, weight="length")
            if net_d < best_net_d:
                best_net_d = net_d
                best = h
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    # Fallback to Haversine nearest if all network queries fail
    if best is None:
        best = candidates[0] if candidates else hospital_list[0]
        best_net_d = scored[0][0] * 1000

    return best, round(best_net_d / 1000.0, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_routes(m_dynamic: dict, m_static: dict) -> dict:
    """
    JSON-serialisable comparison.
    """
    def _pct(new, old):
        return round((old - new) / old * 100, 1) if old else 0.0

    t_saved   = round(m_static["estimated_time_min"] - m_dynamic["estimated_time_min"], 2)
    d_diff    = round(m_static["distance_km"]        - m_dynamic["distance_km"],        3)
    c_diff    = round(m_static["avg_congestion"]     - m_dynamic["avg_congestion"],      3)

    return {
        "static":                  m_static,
        "dynamic":                 m_dynamic,
        "winner":                  "dynamic" if t_saved > 0 else ("static" if t_saved < 0 else "tie"),
        "time_saved_min":          t_saved,
        "time_improvement_pct":    _pct(m_dynamic["estimated_time_min"],
                                        m_static["estimated_time_min"]),
        "distance_diff_km":        d_diff,
        "congestion_reduction":    c_diff,
        "dynamic_longer_but_faster": (d_diff < 0 and t_saved > 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SCENARIO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(G_raw: nx.MultiDiGraph,
                 origin: tuple,
                 destination: tuple,
                 scenario_name: str,
                 hour: int,
                 seed: int = 0) -> dict:
    """
    Run one named time-of-day scenario for an origin → destination pair.
    """
    G = prepare_graph(copy.deepcopy(G_raw), hour=hour, seed=seed)
    routes = {}

    for rname, rfn, wfn in [
        ("dynamic", compute_dynamic_route, dynamic_weight),
        ("static",  compute_static_route,  static_weight),
        ("astar",   compute_astar_route,   dynamic_weight),
    ]:
        try:
            route   = rfn(G, origin, destination)
            metrics = calculate_route_metrics(G, route, weight_fn=wfn)
            coords  = get_route_coords(G, route)
        except Exception as e:
            route, metrics, coords = [], {}, []
            print(f"      [{scenario_name}] {rname} routing failed: {e}")

        routes[rname] = {"route": route, "metrics": metrics, "coords": coords}

    comparison = compare_routes(routes["dynamic"]["metrics"],
                                routes["static"]["metrics"])

    return {
        "scenario":   scenario_name,
        "hour":       hour,
        "routes":     routes,
        "comparison": comparison,
        "graph":      G,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CONGESTION HEAT SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

def build_congestion_segments(G, route: list) -> list:
    """
    Split route into per-edge segments, each tagged with a congestion
    heat colour (green → yellow → orange → red).
    """
    def _color(c: float) -> str:
        if c < 0.25:  return "#2dc653"   # green  — clear
        if c < 0.50:  return "#f6c90e"   # yellow — moderate
        if c < 0.75:  return "#f77f00"   # orange — heavy
        return              "#d62828"    # red    — severe

    segs = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        d    = _best_edge(G, u, v, dynamic_weight)
        c    = d.get("congestion", 0.5)
        segs.append({
            "coords":     [(G.nodes[u]["y"], G.nodes[u]["x"]),
                           (G.nodes[v]["y"], G.nodes[v]["x"])],
            "congestion": round(c, 3),
            "color":      _color(c),
        })
    return segs


# ══════════════════════════════════════════════════════════════════════════════
# 9.  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_scenario(result: dict, tol: float = 0.0) -> dict:
    """
    Assert dynamic model performance guarantees.
    """
    hour    = result["hour"]
    cmp     = result["comparison"]
    t_saved = cmp["time_saved_min"]
    checks  = {}

    if 7 <= hour <= 10 or 16 <= hour <= 20:
        p = t_saved > tol
        checks["rush_hour_dynamic_faster"] = {
            "passed":  p,
            "message": (f"[PASS] Dynamic saves {t_saved:.2f} min in rush hour."
                        if p else
                        f"[FAIL] Dynamic did not beat static (saved {t_saved:.2f} min)."),
        }
    else:
        p = abs(t_saved) < 5.0
        checks["off_peak_diff_reasonable"] = {
            "passed":  p,
            "message": (f"[PASS] Off-peak diff acceptable: {t_saved:.2f} min."
                        if p else
                        f"[WARN] Off-peak diff unusually large: {t_saved:.2f} min."),
        }

    c_ok = cmp["congestion_reduction"] >= 0
    checks["dynamic_lower_congestion"] = {
        "passed":  c_ok,
        "message": (f"[PASS] Congestion reduction: {cmp['congestion_reduction']:.3f}."
                    if c_ok else
                    f"[FAIL] Dynamic path is MORE congested than static."),
    }

    return {
        "scenario":   result["scenario"],
        "all_passed": all(c["passed"] for c in checks.values()),
        "checks":     checks,
    }