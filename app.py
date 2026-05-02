"""
app.py  —  Thrissur Emergency Dispatch Web Platform  v6.1
==========================================================

v6.1 Upgrades (backend):
  - Cache-Control no-store on /simulate & /validate responses
  - Graph pre-cache for all 24 hours (unchanged from v5)
  - Hour/minute slider support (unchanged from v5)
  - Scenario name now includes minutes for display
"""

import copy
import pickle
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from flask import Flask, render_template, request, jsonify
import osmnx as ox

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from routingengine import prepare_graph, SCENARIOS, time_multiplier, haversine
    from dispatchengine import simulate_dispatch, validate_dominance
except ModuleNotFoundError:
    from routingengine import prepare_graph, SCENARIOS, time_multiplier, haversine
    from dispatchengine import simulate_dispatch, validate_dominance

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CACHE
# ══════════════════════════════════════════════════════════════════════════════

CENTER   = (10.5276, 76.2144)
HOUR_MAP = {"morning_rush": 8, "midday": 13, "night": 2}
GRAPH_CACHE = {}  # hour (int) -> enriched graph

print("=" * 60)
print("  THRISSUR EMERGENCY DISPATCH PLATFORM  v6.1")
print("=" * 60)

print("\n[1/3] Loading road network (~30 s)...")
G_raw = ox.graph_from_point(CENTER, dist=7500, network_type="drive")
print(f"      ✅  Nodes: {len(G_raw.nodes)} | Edges: {len(G_raw.edges)}")

PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clustering_data.pkl")
if not os.path.isfile(PKL_PATH):
    PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_data.pkl")

print(f"\n[2/3] Loading clustering data from {PKL_PATH}...")
try:
    with open(PKL_PATH, "rb") as f:
        _data = pickle.load(f)
    kmeans     = _data["kmeans"]
    scaler     = _data["scaler"]
    ambulances = _data["ambulances"]
    hospitals  = _data["hospitals"]
    print(f"      ✅  Ambulances: {len(ambulances)} | Hospitals: {len(hospitals)}")
except FileNotFoundError:
    print("\n  ❌  clustering_data.pkl not found! → Run simulation.py first.")
    sys.exit(1)

print("\n[3/3] Pre-caching graphs for all 24 hours...")
for hour in range(24):
    print(f"      Caching {hour:02d}:00 ...", end=" ", flush=True)
    GRAPH_CACHE[hour] = prepare_graph(copy.deepcopy(G_raw), hour=hour, seed=42)
    print("✅")
print("\n  🚀  System ready at http://localhost:5000\n")
print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SERIALISATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

_STRIP_KEYS = {"route"}

def _clean(obj, _depth=0):
    """Recursively make the dispatch result JSON-serialisable."""
    if _depth > 20:
        return str(obj)
    if isinstance(obj, dict):
        return {k: _clean(v, _depth + 1) for k, v in obj.items() if k not in _STRIP_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_clean(i, _depth + 1) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# v6.1: NO-CACHE RESPONSE HOOK
# ══════════════════════════════════════════════════════════════════════════════

@app.after_request
def add_no_cache_headers(response):
    """Prevent browser/proxy caching of simulation responses."""
    if request.path.startswith('/simulate') or request.path.startswith('/validate'):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/initial_data")
def initial_data():
    """Return pre-deployed ambulances + hospitals for map initialisation."""
    return jsonify({
        "ambulances": _clean(ambulances),
        "hospitals":  _clean(hospitals),
        "center":     list(CENTER),
    })


@app.route("/simulate", methods=["POST"])
def simulate():
    """
    POST body: { lat, lon, hour, minute }
    Supports legacy { scenario } key as fallback.
    """
    body = request.get_json(force=True)

    try:
        lat    = float(body["lat"])
        lon    = float(body["lon"])

        if "hour" in body:
            hour   = min(23, max(0, int(body["hour"])))
            minute = int(body.get("minute", 0))
        else:
            scenario = body.get("scenario", "morning_rush")
            if scenario not in HOUR_MAP:
                return jsonify({"error": f"Unknown scenario: {scenario}"}), 400
            hour   = HOUR_MAP[scenario]
            minute = 0

        G = GRAPH_CACHE.get(hour)
        if G is None:
            print(f"  ⚠️  Cache miss for hour {hour}, computing on-the-fly...")
            G = prepare_graph(copy.deepcopy(G_raw), hour=hour, seed=42)

        result = simulate_dispatch(
            G,
            (lat, lon),
            ambulances,
            hospitals,
            kmeans,
            scaler,
            scenario_name=f"{hour:02d}:{minute:02d}",
            risk_level="Live",
        )

        return jsonify(_clean(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# BATCH VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/validate", methods=["POST"])
def validate():
    """Batch validation — POST body: { count: 20 }"""
    body  = request.get_json(force=True)
    count = int(body.get("count", 20))

    results = []
    import random

    for _ in range(min(count, 50)):
        lat  = random.uniform(10.48, 10.58)
        lon  = random.uniform(76.17, 76.27)
        hour = random.choice(list(GRAPH_CACHE.keys()))
        G    = GRAPH_CACHE[hour]

        result = simulate_dispatch(
            G, (lat, lon), ambulances, hospitals, kmeans, scaler,
            scenario_name=f"{hour:02d}:00", risk_level="Live",
            use_network_ambulance=True,
        )
        results.append(result)

    validation = validate_dominance(results)
    return jsonify({"validation": validation, "samples": len(results)})


# ══════════════════════════════════════════════════════════════════════════════

# Add this line specifically for Vercel to find the app object easily
app = app 

if __name__ == "__main__":
    # This part runs when you run 'python app.py' on your computer
    app.run(debug=False, port=5000, threaded=True)