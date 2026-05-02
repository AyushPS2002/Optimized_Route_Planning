"""
dispatch_engine.py  —  Thrissur Emergency Dispatch Simulation Engine  v5.0
==========================================================================

Upgraded Features:
  - Network-distance ambulance selection (optimal deployment)
  - Enhanced comparison metrics
  - Batch validation support
  - Dispatch origin can be nearest ambulance OR nearest hospital (shortest network)
"""

import time as _time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
import osmnx as ox

# ⚠️ IMPORTANT: All imports from routingengine - haversine is defined there
from routingengine import (
    compute_dynamic_route,
    compute_static_route,
    get_route_coords,
    calculate_route_metrics,
    build_congestion_segments,
    find_nearest_ambulance_network,
    find_nearest_ambulance,
    find_nearest_hospital_network,
    find_nearest_hospital_haversine,
    dynamic_weight,
    static_weight,
    haversine,
    SCENARIOS,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ON_SCENE_DELAY_MIN = 4.0     # stabilisation + patient loading (minutes)

RISK_LABELS = {
    "High":   "🔴 High Risk",
    "Medium": "🟠 Medium Risk",
    "Low":    "🟡 Low Risk",
}

SCENARIO_LABELS = {
    "morning_rush": "🕗 Morning Rush (08:00)",
    "midday":       "☀️  Midday (13:00)",
    "night":        "🌙 Night (02:00)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ZONE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_zone(kmeans_model, scaler, accident_loc: tuple) -> int:
    """
    Classify accident location into one of the 6 KMeans zones.
    """
    scaled = scaler.transform([[accident_loc[0], accident_loc[1]]])
    return int(kmeans_model.predict(scaled)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UPGRADED SINGLE DISPATCH SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_dispatch(G: nx.MultiDiGraph,
                      accident_loc: tuple,
                      ambulance_locations: list,
                      hospital_locations: list,
                      kmeans_model,
                      scaler,
                      scenario_name: str = "morning_rush",
                      risk_level: str = "High",
                      use_network_ambulance: bool = True) -> dict:
    """
    Simulate the complete accident → dispatch → hospital cycle.
    UPGRADED: Dispatch origin can be the nearest ambulance OR the nearest hospital,
              whichever gives the shortest network distance to the accident.
    """
    t0 = _time.perf_counter()
    result = {
        "scenario":    scenario_name,
        "risk_level":  risk_level,
        "accident_loc": accident_loc,
    }

    # ── 1. Zone detection ─────────────────────────────────────────────────────
    zone_id = detect_zone(kmeans_model, scaler, accident_loc)
    result["zone_id"] = zone_id

    # ── 2. Find nearest ambulance and nearest hospital by network distance ───
    nearest_amb, amb_net_km = find_nearest_ambulance_network(G, accident_loc, ambulance_locations)
    nearest_hosp, hosp_net_km = find_nearest_hospital_network(G, accident_loc, hospital_locations)

    # Choose the closer origin
    if amb_net_km <= hosp_net_km:
        # Ambulance is closer
        origin = nearest_amb
        origin_net_km = amb_net_km
        origin_type = "ambulance"
        origin_label = nearest_amb.get("label", "Ambulance")
    else:
        # Hospital is closer – use it as dispatch origin
        origin = nearest_hosp
        origin_net_km = hosp_net_km
        origin_type = "hospital"
        # Create a label for the hospital origin
        origin_label = f"Hospital: {nearest_hosp['name']}"

    origin_loc = (origin["lat"], origin["lon"])
    result["dispatch_origin"] = {
        **origin,
        "type": origin_type,
        "network_distance_km": origin_net_km,
        "straight_line_km": haversine(accident_loc[0], accident_loc[1],
                                       origin["lat"], origin["lon"]),
        "label": origin_label,
    }

    # For backward compatibility, also keep an "ambulance" field
    result["ambulance"] = result["dispatch_origin"]

    # ── 3a. DYNAMIC model hospital — nearest by ROAD-NETWORK distance ─────────
    nearest_hosp_dyn, hosp_net_km = find_nearest_hospital_network(
        G, accident_loc, hospital_locations
    )
    hosp_dyn_loc = (nearest_hosp_dyn["lat"], nearest_hosp_dyn["lon"])
    result["hospital_dynamic"] = {
        **nearest_hosp_dyn,
        "network_distance_km": hosp_net_km,
        "selection_method": "network_distance",
    }

    # ── 3b. STATIC model hospital — nearest by HAVERSINE straight-line ────────
    nearest_hosp_stat, hosp_hav_km = find_nearest_hospital_haversine(
        accident_loc, hospital_locations
    )
    hosp_stat_loc = (nearest_hosp_stat["lat"], nearest_hosp_stat["lon"])
    result["hospital_static"] = {
        **nearest_hosp_stat,
        "haversine_distance_km": hosp_hav_km,
        "selection_method": "haversine",
    }

    # Backwards-compatible alias
    result["hospital"] = result["hospital_dynamic"]

    # ══════════════════════════════════════════════════════════════════════════
    # DYNAMIC MODEL: routing throughout
    # ══════════════════════════════════════════════════════════════════════════
    dyn = {}

    # Leg 1: origin (ambulance or hospital) → accident
    try:
        r1d       = compute_dynamic_route(G, origin_loc, accident_loc)
        m1d       = calculate_route_metrics(G, r1d, weight_fn=dynamic_weight)
        c1d       = get_route_coords(G, r1d)
        segs1d    = build_congestion_segments(G, r1d)
    except Exception as e:
        r1d, m1d, c1d, segs1d = [], _empty_metrics(), [], []
        print(f"  [WARN] Dynamic dispatch route failed: {e}")

    # Leg 2: accident → network-nearest hospital (dynamic)
    try:
        r2d       = compute_dynamic_route(G, accident_loc, hosp_dyn_loc)
        m2d       = calculate_route_metrics(G, r2d, weight_fn=dynamic_weight)
        c2d       = get_route_coords(G, r2d)
        segs2d    = build_congestion_segments(G, r2d)
    except Exception as e:
        r2d, m2d, c2d, segs2d = [], _empty_metrics(), [], []
        print(f"  [WARN] Dynamic hospital route failed: {e}")

    dyn_total = round(
        m1d["estimated_time_min"] + ON_SCENE_DELAY_MIN + m2d["estimated_time_min"], 2
    )
    dyn = {
        "to_accident": {**m1d, "coords": c1d, "route": r1d, "cong_segs": segs1d},
        "to_hospital": {**m2d, "coords": c2d, "route": r2d, "cong_segs": segs2d},
        "on_scene_delay_min": ON_SCENE_DELAY_MIN,
        "total_response_min": dyn_total,
        "total_distance_km":  round(
            m1d["distance_km"] + m2d["distance_km"], 3
        ),
        "avg_congestion": round(
            (m1d["avg_congestion"] + m2d["avg_congestion"]) / 2, 3
        ),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE MODEL: static routing throughout
    # ══════════════════════════════════════════════════════════════════════════
    stat = {}

    # Leg 1: haversine-nearest hospital → accident (static)
    try:
        r1s    = compute_static_route(G, hosp_stat_loc, accident_loc)
        m1s    = calculate_route_metrics(G, r1s, weight_fn=static_weight)
        c1s    = get_route_coords(G, r1s)
    except Exception as e:
        r1s, m1s, c1s = [], _empty_metrics(), []
        print(f"  [WARN] Static dispatch route failed: {e}")

    # Leg 2: accident → same hospital (static)
    try:
        r2s    = compute_static_route(G, accident_loc, hosp_stat_loc)
        m2s    = calculate_route_metrics(G, r2s, weight_fn=static_weight)
        c2s    = get_route_coords(G, r2s)
    except Exception as e:
        r2s, m2s, c2s = [], _empty_metrics(), []
        print(f"  [WARN] Static hospital route failed: {e}")

    stat_total = round(
        m1s["estimated_time_min"] + ON_SCENE_DELAY_MIN + m2s["estimated_time_min"], 2
    )
    stat = {
        "to_accident": {**m1s, "coords": c1s, "route": r1s},
        "to_hospital": {**m2s, "coords": c2s, "route": r2s},
        "on_scene_delay_min": ON_SCENE_DELAY_MIN,
        "total_response_min": stat_total,
        "total_distance_km":  round(
            m1s["distance_km"] + m2s["distance_km"], 3
        ),
        "avg_congestion": round(
            (m1s["avg_congestion"] + m2s["avg_congestion"]) / 2, 3
        ),
    }

    # ── Comparison ────────────────────────────────────────────────────────────
    time_saved   = round(stat_total - dyn_total, 2)
    time_pct     = round((stat_total - dyn_total) / stat_total * 100, 1) if stat_total else 0.0
    dist_diff    = round(stat["total_distance_km"] - dyn["total_distance_km"], 3)
    cong_diff    = round(stat["avg_congestion"]    - dyn["avg_congestion"],    3)
    winner       = "dynamic" if time_saved > 0 else ("static" if time_saved < 0 else "tie")

    comparison = {
        "winner":                winner,
        "time_saved_min":        time_saved,
        "time_improvement_pct":  time_pct,
        "distance_diff_km":      dist_diff,
        "congestion_reduction":  cong_diff,
        "dynamic_longer_but_faster": dist_diff < 0 and time_saved > 0,
    }

    result.update({
        "dynamic":    dyn,
        "static":     stat,
        "comparison": comparison,
        "compute_time_s": round(_time.perf_counter() - t0, 2),
    })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MULTI-SCENARIO BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_scenario_dispatch(G_raw: nx.MultiDiGraph,
                                 accident_cases: list,
                                 ambulance_locations: list,
                                 hospital_locations: list,
                                 kmeans_model,
                                 scaler,
                                 scenarios: dict = None,
                                 use_network_ambulance: bool = True) -> list:
    """
    Run the full dispatch simulation for every (accident, scenario) combination.
    """
    import copy
    from routingengine import prepare_graph

    if scenarios is None:
        scenarios = SCENARIOS

    all_results = []

    for case in accident_cases:
        acc_loc    = case["accident_loc"]
        risk       = case["risk_level"]
        case_label = case.get("label", f"Accident ({risk})")

        for sc_name, sc_hour in scenarios.items():
            print(f"  [{sc_name}] {case_label} ({risk}) ...", end=" ", flush=True)
            t0 = _time.perf_counter()

            G = prepare_graph(copy.deepcopy(G_raw), hour=sc_hour, seed=42)

            res = simulate_dispatch(
                G, acc_loc,
                ambulance_locations, hospital_locations,
                kmeans_model, scaler,
                scenario_name=sc_name,
                risk_level=risk,
                use_network_ambulance=use_network_ambulance,
            )
            res["case_label"] = case_label
            res["zone_label"]  = case.get("zone_label", "")
            res["G_enriched"]  = G

            elapsed = _time.perf_counter() - t0
            cmp = res["comparison"]
            print(f"done ({elapsed:.1f}s) | winner={cmp['winner']} | "
                  f"saved={cmp['time_saved_min']:.1f} min")

            all_results.append(res)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONSOLE SUMMARY FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_dispatch_summary(result: dict) -> str:
    """
    Return a formatted multi-line string summarising one dispatch result.
    """
    dyn  = result["dynamic"]
    stat = result["static"]
    cmp  = result["comparison"]
    amb  = result["dispatch_origin"]   # use dispatch_origin
    hosp_dyn  = result.get("hospital_dynamic", result["hospital"])
    hosp_stat = result.get("hospital_static",  result["hospital"])
    sc   = SCENARIO_LABELS.get(result["scenario"], result["scenario"])
    rl   = RISK_LABELS.get(result["risk_level"], result["risk_level"])
    w    = "🚀 DYNAMIC" if cmp["winner"] == "dynamic" else "📏 STATIC"

    sep = "─" * 60
    lines = [
        sep,
        f"  DISPATCH RESULT  |  {sc}  |  {rl}",
        sep,
        f"  Zone            : Zone {result['zone_id']}  ({result.get('zone_label','')})",
        f"  Accident        : {result['accident_loc'][0]:.5f}, {result['accident_loc'][1]:.5f}",
        f"  [DYN] Origin    : {amb['label']} "
        f"(network: {amb['network_distance_km']} km, straight: {amb['straight_line_km']} km)",
        f"  [DYN] Hospital  : {hosp_dyn['name']} "
        f"({hosp_dyn['network_distance_km']} km by road — network selection)",
        f"  [STA] Hospital  : {hosp_stat['name']} "
        f"({hosp_stat.get('haversine_distance_km', '?')} km straight-line — Haversine selection)",
        "",
        f"  {'Metric':<26}  {'Dynamic':>10}  {'Static':>10}",
        f"  {'':─<26}  {'':─>10}  {'':─>10}",
        f"  {'Dispatch distance (km)':<26}  {dyn['to_accident']['distance_km']:>10.3f}  {stat['to_accident']['distance_km']:>10.3f}",
        f"  {'Dispatch time (min)':<26}  {dyn['to_accident']['estimated_time_min']:>10.2f}  {stat['to_accident']['estimated_time_min']:>10.2f}",
        f"  {'Dispatch congestion':<26}  {dyn['to_accident']['avg_congestion']:>10.3f}  {stat['to_accident']['avg_congestion']:>10.3f}",
        f"  {'Transport distance (km)':<26}  {dyn['to_hospital']['distance_km']:>10.3f}  {stat['to_hospital']['distance_km']:>10.3f}",
        f"  {'Transport time (min)':<26}  {dyn['to_hospital']['estimated_time_min']:>10.2f}  {stat['to_hospital']['estimated_time_min']:>10.2f}",
        f"  {'Transport congestion':<26}  {dyn['to_hospital']['avg_congestion']:>10.3f}  {stat['to_hospital']['avg_congestion']:>10.3f}",
        f"  {'On-scene delay (min)':<26}  {ON_SCENE_DELAY_MIN:>10.1f}  {ON_SCENE_DELAY_MIN:>10.1f}",
        f"  {'':─<26}  {'':─>10}  {'':─>10}",
        f"  {'TOTAL RESPONSE (min)':<26}  {dyn['total_response_min']:>10.2f}  {stat['total_response_min']:>10.2f}",
        f"  {'Total distance (km)':<26}  {dyn['total_distance_km']:>10.3f}  {stat['total_distance_km']:>10.3f}",
        f"  {'Avg congestion':<26}  {dyn['avg_congestion']:>10.3f}  {stat['avg_congestion']:>10.3f}",
        "",
        f"  Winner       : {w}",
        f"  Time saved   : {cmp['time_saved_min']} min  ({cmp['time_improvement_pct']}%)",
        f"  Dist diff    : {cmp['distance_diff_km']:+.3f} km  "
        f"({'dynamic longer but faster' if cmp['dynamic_longer_but_faster'] else 'dynamic also shorter'})",
        f"  Cong reduced : {cmp['congestion_reduction']:.3f}",
        sep,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BATCH VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dominance(results: list, threshold_pct: float = 95.0) -> dict:
    """
    Validate that dynamic wins in at least threshold_pct of scenarios.
    Used to verify system maturity.
    """
    total = len(results)
    wins = sum(1 for r in results if r["comparison"]["winner"] == "dynamic")
    win_pct = (wins / total) * 100 if total > 0 else 0
    
    return {
        "total_scenarios": total,
        "dynamic_wins": wins,
        "win_percentage": round(win_pct, 1),
        "passes_threshold": win_pct >= threshold_pct,
        "avg_time_saved": np.mean([r["comparison"]["time_saved_min"] for r in results]) if results else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _empty_metrics() -> dict:
    return {
        "distance_km": 0.0, "avg_congestion": 0.0,
        "avg_reliability": 0.0, "estimated_time_min": 0.0,
        "num_edges": 0, "avg_speed_kmh": 0.0, "route_cost": 0.0,
    }