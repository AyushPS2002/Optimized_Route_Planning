"""
simulation_test.py  —  Thrissur Emergency Routing Simulation  v3.0
==================================================================

Pipeline
--------
  PHASE 0  Clustering          (same seeds as main1.py → identical data)
  PHASE 1  Single Accident     Full dispatch cycle — one High-risk accident
  PHASE 2  Multi-Scenario      9 cases (High / Medium / Low) × 3 time slots
  PHASE 3  Visualisation       Narrative Folium map

Map layers (toggleable)
-----------------------
  🗺  Zone Boundaries
  🔴  High / 🟠 Medium / 🟡 Low accidents
  🚑  Standby ambulances
  🏥  Hospitals
  🌡  Heatmap

  Per dispatch scenario (Phase 1 — main narrative):
  ──  Ambulance → Accident  (red   solid = dynamic, blue dashed = static)
  ──  Accident  → Hospital  (purple solid = dynamic, teal  dashed = static)
  🌡  Congestion heat overlay along dynamic dispatch route

  Phase 2 summary routes (morning rush only, lighter weight)

Run
---
  python simulation_test.py
"""

import copy
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import folium
from folium.plugins import MiniMap, HeatMap
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import osmnx as ox

from routingengine import (
    prepare_graph, SCENARIOS,
    get_route_coords,
    build_congestion_segments,
    dynamic_weight, static_weight,
    haversine,
)
# Import from dispatchengine.py (no underscore — matches filename in project)
try:
    from dispatchengine import (
        simulate_dispatch,
        run_multi_scenario_dispatch,
        format_dispatch_summary,
        find_nearest_hospital_network,
        ON_SCENE_DELAY_MIN,
        RISK_LABELS,
        SCENARIO_LABELS,
    )
except ModuleNotFoundError:
    from dispatchengine import (   # fallback for underscore variant
        simulate_dispatch,
        run_multi_scenario_dispatch,
        format_dispatch_summary,
        find_nearest_hospital_network,
        ON_SCENE_DELAY_MIN,
        RISK_LABELS,
        SCENARIO_LABELS,
    )

SEP  = "═" * 65
SEP2 = "─" * 65

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — CLUSTERING  (identical to main1.py)
# ══════════════════════════════════════════════════════════════════════════════

print(SEP)
print("  THRISSUR EMERGENCY DISPATCH SIMULATION  v3.0")
print(SEP)

# ── P0.1  Road network ────────────────────────────────────────────────────────
print("\n[0/6] Loading road network...")
CENTER  = (10.5276, 76.2144)
G_raw   = ox.graph_from_point(CENTER, dist=7500, network_type="drive")
nodes   = list(G_raw.nodes(data=True))
print(f"      Nodes: {len(G_raw.nodes)} | Edges: {len(G_raw.edges)}")

# ── P0.2  Accident generation ─────────────────────────────────────────────────
print("[1/6] Generating accidents...")
HOTSPOT_SEEDS = [
    (10.5210, 76.2120), (10.5300, 76.2200), (10.5350, 76.2050),
    (10.5150, 76.2300), (10.5420, 76.2180), (10.5080, 76.2080),
    (10.5270, 76.2400), (10.5480, 76.2320), (10.5050, 76.2250),
    (10.5600, 76.2100),
]
random.seed(42); np.random.seed(42)
junction_nodes = [n for n in nodes if G_raw.degree(n[0]) >= 3]
acc = []
for _ in range(600):
    s = random.choice(HOTSPOT_SEEDS)
    acc.append([s[0] + np.random.normal(0, 0.008), s[1] + np.random.normal(0, 0.008)])
for _ in range(300):
    n = random.choice(junction_nodes)
    acc.append([n[1]["y"] + np.random.normal(0, 0.001), n[1]["x"] + np.random.normal(0, 0.001)])
for _ in range(100):
    n = random.choice(nodes)
    acc.append([n[1]["y"], n[1]["x"]])

df = pd.DataFrame(acc, columns=["lat", "lon"])
df = df[(df["lat"].between(10.48, 10.58)) &
        (df["lon"].between(76.17, 76.27))].reset_index(drop=True)
print(f"      Accidents: {len(df)}")

# ── P0.3  KMeans ──────────────────────────────────────────────────────────────
print("[2/6] KMeans zoning...")
scaler_g = StandardScaler()
coords_s = scaler_g.fit_transform(df[["lat", "lon"]])
kmeans   = KMeans(n_clusters=6, random_state=42, n_init=15, max_iter=500)
df["zone"] = kmeans.fit_predict(coords_s).astype(int)

ZONE_COLORS = {0:"#b5d5f5", 1:"#fde8a8", 2:"#c3e6cb",
               3:"#f5c6cb", 4:"#d4b8e0", 5:"#b2e0d9"}
ZONE_BORDER = {0:"#2a7abf", 1:"#c87f00", 2:"#2e7d4f",
               3:"#b5434a", 4:"#7b4fa6", 5:"#1f7a6e"}
ZONE_LABELS = {
    0:"Zone N (North)", 1:"Zone NE (North-East)", 2:"Zone E (East)",
    3:"Zone C (Central)", 4:"Zone W (West)", 5:"Zone S (South)",
}

# ── P0.4  DBSCAN risk ─────────────────────────────────────────────────────────
print("[3/6] DBSCAN risk classification...")
DBSCAN_EPS=0.22; DBSCAN_MIN=4; HIGH_PCT=0.20; MED_PCT=0.30
df["risk"]="Low"; df["db_cluster"]=-1
cluster_centroids=[]

for zid in sorted(df["zone"].unique()):
    mask = df["zone"]==zid
    zpts = df.loc[mask, ["lat","lon"]].copy()
    if len(zpts)<DBSCAN_MIN: continue
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(
        StandardScaler().fit_transform(zpts))
    df.loc[mask,"db_cluster"] = labels
    zcl=[]
    for cid in set(labels):
        if cid==-1: continue
        cm=mask&(df["db_cluster"]==cid)
        zcl.append({"cid":cid,"mask":cm,"sz":int(cm.sum())})
    zcl.sort(key=lambda x:x["sz"],reverse=True)
    if not zcl: continue
    nc=len(zcl); nh=max(1,round(nc*HIGH_PCT)); nm=max(1,round(nc*MED_PCT))
    for rank,cl in enumerate(zcl):
        risk="High" if rank<nh else ("Medium" if rank<nh+nm else "Low")
        df.loc[cl["mask"],"risk"]=risk
        cluster_centroids.append({
            "zone":zid,"cluster_id":cl["cid"],"risk":risk,
            "lat":df.loc[cl["mask"],"lat"].mean(),
            "lon":df.loc[cl["mask"],"lon"].mean(),
            "size":cl["sz"],
        })

rc=df["risk"].value_counts()
print(f"      High:{rc.get('High',0)}  Medium:{rc.get('Medium',0)}  Low:{rc.get('Low',0)}")

# ── P0.5  Ambulances ──────────────────────────────────────────────────────────
print("[4/6] Placing ambulances...")
AMBU_COUNT={"High":5,"Medium":3,"Low":1}; SPREAD=0.003; rng_a=np.random.default_rng(7)
vehicle_locations=[]
for cc in cluster_centroids:
    n=AMBU_COUNT[cc["risk"]]
    for i in range(n):
        ang=(2*np.pi*i)/max(n,1)
        vehicle_locations.append({
            "lat":  cc["lat"]+SPREAD*np.sin(ang)*rng_a.uniform(0.6,1.1),
            "lon":  cc["lon"]+SPREAD*np.cos(ang)*rng_a.uniform(0.6,1.1),
            "zone": cc["zone"], "cluster_id":cc["cluster_id"], "risk":cc["risk"],
            "label":f"{ZONE_LABELS[cc['zone']]} | {cc['risk']} cluster (n={cc['size']})",
            "id":   len(vehicle_locations),
        })
print(f"      Ambulances: {len(vehicle_locations)}")

# ── P0.6  Hospitals ───────────────────────────────────────────────────────────
print("[5/6] Fetching hospitals...")
hosp_gdf = ox.features_from_point(CENTER, tags={"amenity":"hospital"}, dist=7500)
hospital_locations=[]
for _,row in hosp_gdf.iterrows():
    geom=row.geometry
    if geom is None: continue
    nm=row.get("name","Hospital")
    hospital_locations.append({
        "lat":geom.centroid.y,"lon":geom.centroid.x,
        "name":nm if isinstance(nm,str) else "Hospital",
    })
print(f"      Hospitals: {len(hospital_locations)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SINGLE ACCIDENT FULL DISPATCH  (morning rush, High-risk)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Running dispatch simulations...\n{SEP2}")
print("  PHASE 1 — Single accident full dispatch cycle")
print(SEP2)

# Select most central High-risk accident (centroid of all High points)
high_df   = df[df["risk"]=="High"]
h_clat, h_clon = high_df["lat"].mean(), high_df["lon"].mean()
high_df   = high_df.copy()
high_df["dc"] = ((high_df["lat"]-h_clat)**2+(high_df["lon"]-h_clon)**2)**0.5
p1_row    = high_df.nsmallest(1,"dc").iloc[0]
P1_ACC    = (p1_row["lat"], p1_row["lon"])

# Enrich graph for morning rush
G_rush    = prepare_graph(copy.deepcopy(G_raw), hour=8, seed=42)

print(f"  Accident : {P1_ACC[0]:.5f}, {P1_ACC[1]:.5f}  (High-risk, central)")
print(f"  Scenario : Morning Rush (08:00)\n")

p1_result = simulate_dispatch(
    G_rush, P1_ACC,
    vehicle_locations, hospital_locations,
    kmeans, scaler_g,
    scenario_name="morning_rush",
    risk_level="High",
)

print(format_dispatch_summary(p1_result))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MULTI-SCENARIO BATCH  (High / Medium / Low × 3 time slots)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP2}")
print("  PHASE 2 — Multi-scenario batch simulation")
print(SEP2)

# Select one representative accident per risk level
accident_cases = []
for risk_lvl in ["High", "Medium", "Low"]:
    risk_pts = df[df["risk"]==risk_lvl]
    if risk_pts.empty: continue
    clat, clon = risk_pts["lat"].mean(), risk_pts["lon"].mean()
    risk_pts   = risk_pts.copy()
    risk_pts["dc"] = ((risk_pts["lat"]-clat)**2+(risk_pts["lon"]-clon)**2)**0.5
    row = risk_pts.nsmallest(1,"dc").iloc[0]
    zone_id = int(row["zone"])
    accident_cases.append({
        "accident_loc": (row["lat"], row["lon"]),
        "risk_level":   risk_lvl,
        "label":        f"{risk_lvl}-risk accident",
        "zone_id":      zone_id,
        "zone_label":   ZONE_LABELS[zone_id],
    })

print(f"  Running {len(accident_cases)} accidents × {len(SCENARIOS)} scenarios "
      f"= {len(accident_cases)*len(SCENARIOS)} total dispatches\n")

p2_results = run_multi_scenario_dispatch(
    G_raw, accident_cases,
    vehicle_locations, hospital_locations,
    kmeans, scaler_g,
    scenarios=SCENARIOS,
)

# ── Console summary table ─────────────────────────────────────────────────────
print(f"\n{SEP2}")
print(f"  PHASE 2 RESULTS SUMMARY")
print(SEP2)
header = (f"  {'Risk':<8} {'Scenario':<16} {'Hr':>3}  "
          f"{'Dyn tot':>8} {'Sta tot':>8}  "
          f"{'Dyn cong':>9} {'Sta cong':>9}  "
          f"{'Saved':>6}  {'Winner':<8}")
print(header)
print("  " + "─"*85)
for res in p2_results:
    dyn=res["dynamic"]; stat=res["static"]; cmp=res["comparison"]
    win_sym = "✅" if cmp["winner"]=="dynamic" else "❌"
    print(f"  {res['risk_level']:<8} {res['scenario']:<16} {res['scenario'][:1]:>3}  "
          f"{dyn['total_response_min']:>8.2f} {stat['total_response_min']:>8.2f}  "
          f"{dyn['avg_congestion']:>9.3f} {stat['avg_congestion']:>9.3f}  "
          f"{cmp['time_saved_min']:>6.2f}  {cmp['winner']:<8} {win_sym}")

# Aggregate stats
n_wins = sum(1 for r in p2_results if r["comparison"]["winner"]=="dynamic")
avg_saved = np.mean([r["comparison"]["time_saved_min"] for r in p2_results])
print(f"\n  Dynamic wins: {n_wins}/{len(p2_results)} scenarios")
print(f"  Avg time saved: {avg_saved:.2f} min across all scenarios")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP2}")
print("  PHASE 3 — Building narrative map...")

RISK_FILL    = {"Low":"#f6c90e","Medium":"#f77f00","High":"#d62828"}
RISK_RADIUS  = {"Low":4,"Medium":6,"High":9}
RISK_OPACITY = {"Low":0.50,"Medium":0.70,"High":0.88}
AMBU_ICON    = {"High":"red","Medium":"orange","Low":"beige"}

m = folium.Map(location=CENTER, zoom_start=13, tiles="CartoDB positron")
MiniMap(toggle_display=True, position="bottomright").add_to(m)

# ── Layer groups ──────────────────────────────────────────────────────────────
lg = {
    # Cluster layers
    "zones":    folium.FeatureGroup(name="🗺️ Zone Boundaries",            show=True),
    "high":     folium.FeatureGroup(name="🔴 High Risk Accidents",         show=True),
    "medium":   folium.FeatureGroup(name="🟠 Medium Risk Accidents",       show=True),
    "low":      folium.FeatureGroup(name="🟡 Low Risk Accidents",          show=False),
    "ambu":     folium.FeatureGroup(name="🚑 Standby Ambulances",          show=True),
    "hosp":     folium.FeatureGroup(name="🏥 Hospitals",                   show=True),
    "heat_acc": folium.FeatureGroup(name="🌡️ Accident Heatmap",            show=False),

    # Phase 1 — main narrative routes
    "p1_dyn_disp":  folium.FeatureGroup(name="🔴 [P1] Dynamic: Amb→Accident",     show=True),
    "p1_stat_disp": folium.FeatureGroup(name="🔵 [P1] Static:  Hosp→Accident",    show=True),
    "p1_dyn_hosp":  folium.FeatureGroup(name="🟣 [P1] Dynamic: Accident→Hospital", show=True),
    "p1_stat_hosp": folium.FeatureGroup(name="🩵 [P1] Static:  Accident→Hospital", show=True),
    "p1_cong":      folium.FeatureGroup(name="🌡️ [P1] Congestion Heat (Dynamic)",  show=False),
    "p1_markers":   folium.FeatureGroup(name="📍 [P1] Dispatch Markers",            show=True),

    # Phase 2 — multi-scenario summary (morning rush)
    "p2_rush_dyn":  folium.FeatureGroup(name="🔴 [P2] Morning Rush Dynamic",       show=False),
    "p2_rush_stat": folium.FeatureGroup(name="🔵 [P2] Morning Rush Static",        show=False),
    "p2_night_dyn": folium.FeatureGroup(name="🌙 [P2] Night Dynamic",              show=False),
}

# ── Zone convex hulls ─────────────────────────────────────────────────────────
for zid in sorted(df["zone"].unique()):
    zpts=df[df["zone"]==zid][["lat","lon"]].values
    if len(zpts)<4: continue
    try:
        hull=ConvexHull(zpts)
        hc=zpts[hull.vertices].tolist()+[zpts[hull.vertices[0]].tolist()]
        folium.Polygon(
            [[p[0],p[1]] for p in hc],
            color=ZONE_BORDER[zid],weight=2.5,
            fill=True,fill_color=ZONE_COLORS[zid],fill_opacity=0.20,
            dash_array="6 4",
            tooltip=folium.Tooltip(
                f"<b>{ZONE_LABELS[zid]}</b><br>"
                f"H:{len(df[(df['zone']==zid)&(df['risk']=='High')])} "
                f"M:{len(df[(df['zone']==zid)&(df['risk']=='Medium')])} "
                f"L:{len(df[(df['zone']==zid)&(df['risk']=='Low')])}"
            )
        ).add_to(lg["zones"])
        clat,clon=zpts[:,0].mean(),zpts[:,1].mean()
        folium.Marker([clat,clon],icon=folium.DivIcon(
            html=(f'<div style="background:{ZONE_COLORS[zid]};border:2px solid {ZONE_BORDER[zid]};'
                  f'color:{ZONE_BORDER[zid]};font-size:10px;font-weight:bold;'
                  f'padding:2px 6px;border-radius:10px;white-space:nowrap;'
                  f'box-shadow:0 1px 4px rgba(0,0,0,0.25)">{ZONE_LABELS[zid]}</div>'),
            icon_size=(140,24),icon_anchor=(70,12)
        )).add_to(lg["zones"])
    except Exception:
        pass

# ── Heatmap ───────────────────────────────────────────────────────────────────
HeatMap([[r["lat"],r["lon"]] for _,r in df.iterrows()],
        radius=14,blur=18,max_zoom=14,
        gradient={"0.3":"blue","0.55":"lime","0.75":"orange","1.0":"red"}
        ).add_to(lg["heat_acc"])

# ── Accident dots ─────────────────────────────────────────────────────────────
risk_lg={"High":"high","Medium":"medium","Low":"low"}
for _,row in df.iterrows():
    fill=RISK_FILL[row["risk"]]
    folium.CircleMarker(
        [row["lat"],row["lon"]],
        radius=RISK_RADIUS[row["risk"]],color="#444",fill=True,
        fill_color=fill,fill_opacity=RISK_OPACITY[row["risk"]],weight=0.5,
        tooltip=folium.Tooltip(
            f"<b>{ZONE_LABELS[row['zone']]}</b><br>"
            f"Risk: <b style='color:{fill}'>{row['risk']}</b>"
        )
    ).add_to(lg[risk_lg[row["risk"]]])

# ── Standby ambulances ────────────────────────────────────────────────────────
for v in vehicle_locations:
    folium.Marker(
        [v["lat"],v["lon"]],
        icon=folium.Icon(color=AMBU_ICON[v["risk"]],icon="ambulance",prefix="fa"),
        tooltip=folium.Tooltip(f"<b>🚑 Standby</b><br>{v['label']}")
    ).add_to(lg["ambu"])

# ── Hospitals ─────────────────────────────────────────────────────────────────
for h in hospital_locations:
    folium.Marker(
        [h["lat"],h["lon"]],
        icon=folium.DivIcon(
            html=('<div style="font-size:13px;color:white;background:crimson;'
                  'border-radius:50%;width:20px;height:20px;display:flex;'
                  'align-items:center;justify-content:center;'
                  'border:2px solid white;box-shadow:0 0 4px rgba(0,0,0,0.4);'
                  'font-weight:bold">+</div>'),
            icon_size=(22,22),icon_anchor=(11,11)
        ),
        tooltip=folium.Tooltip(f"🏥 <b>{h['name']}</b>")
    ).add_to(lg["hosp"])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — NARRATIVE ROUTES ON MAP
# ══════════════════════════════════════════════════════════════════════════════

dyn  = p1_result["dynamic"]
stat = p1_result["static"]
cmp  = p1_result["comparison"]
amb  = p1_result["ambulance"]
hosp = p1_result["hospital"]

# ── Dynamic: ambulance → accident  (RED solid) ────────────────────────────────
if dyn["to_accident"]["coords"]:
    md1=dyn["to_accident"]
    folium.PolyLine(
        md1["coords"],color="#d62828",weight=7,opacity=0.90,
        tooltip=folium.Tooltip(
            f"<b>🔴 [P1 Dynamic] Ambulance → Accident</b><br>"
            f"Distance: {md1['distance_km']} km | "
            f"<b>Time: {md1['estimated_time_min']} min</b><br>"
            f"Avg Congestion: {md1['avg_congestion']} | "
            f"Avg Speed: {md1['avg_speed_kmh']} km/h"
        )
    ).add_to(lg["p1_dyn_disp"])

# ── Static: hospital → accident  (BLUE dashed) ───────────────────────────────
if stat["to_accident"]["coords"]:
    ms1=stat["to_accident"]
    folium.PolyLine(
        ms1["coords"],color="#1a6faf",weight=5,opacity=0.75,
        dash_array="9 5",
        tooltip=folium.Tooltip(
            f"<b>🔵 [P1 Static] Hospital → Accident</b><br>"
            f"Distance: {ms1['distance_km']} km | "
            f"Time: {ms1['estimated_time_min']} min<br>"
            f"Avg Congestion: {ms1['avg_congestion']}"
        )
    ).add_to(lg["p1_stat_disp"])

# ── Dynamic: accident → hospital  (PURPLE solid) ─────────────────────────────
if dyn["to_hospital"]["coords"]:
    md2=dyn["to_hospital"]
    folium.PolyLine(
        md2["coords"],color="#8e44ad",weight=7,opacity=0.88,
        tooltip=folium.Tooltip(
            f"<b>🟣 [P1 Dynamic] Accident → Hospital</b><br>"
            f"To: {hosp['name']}<br>"
            f"Distance: {md2['distance_km']} km | "
            f"<b>Time: {md2['estimated_time_min']} min</b><br>"
            f"Avg Congestion: {md2['avg_congestion']}"
        )
    ).add_to(lg["p1_dyn_hosp"])

# ── Static: accident → hospital  (TEAL dashed) ───────────────────────────────
if stat["to_hospital"]["coords"]:
    ms2=stat["to_hospital"]
    folium.PolyLine(
        ms2["coords"],color="#1f7a6e",weight=5,opacity=0.72,
        dash_array="9 5",
        tooltip=folium.Tooltip(
            f"<b>🩵 [P1 Static] Accident → Hospital</b><br>"
            f"To: {hosp['name']}<br>"
            f"Distance: {ms2['distance_km']} km | "
            f"Time: {ms2['estimated_time_min']} min"
        )
    ).add_to(lg["p1_stat_hosp"])

# ── Congestion heat along dynamic dispatch route ──────────────────────────────
for seg in dyn["to_accident"].get("cong_segs", []):
    folium.PolyLine(
        seg["coords"],color=seg["color"],weight=10,opacity=0.50,
        tooltip=folium.Tooltip(f"Congestion: {seg['congestion']:.2f}")
    ).add_to(lg["p1_cong"])

# ── P1 Markers ────────────────────────────────────────────────────────────────
# Accident scene
folium.CircleMarker(
    P1_ACC,radius=22,color="#d62828",fill=False,weight=3,
    opacity=0.6,dash_array="4 3"
).add_to(lg["p1_markers"])

folium.Marker(
    P1_ACC,
    icon=folium.DivIcon(
        html=('<div style="background:#d62828;color:white;border-radius:50%;'
              'width:32px;height:32px;display:flex;align-items:center;'
              'justify-content:center;font-size:18px;border:3px solid white;'
              'box-shadow:0 0 8px rgba(0,0,0,0.6)">⚠</div>'),
        icon_size=(32,32),icon_anchor=(16,16)
    ),
    tooltip=folium.Tooltip(
        f"<b>⚠️ ACCIDENT SCENE — Phase 1</b><br>"
        f"High-risk | Zone {p1_result['zone_id']}<br>"
        f"Nearest amb: {amb['straight_line_km']} km<br>"
        f"Nearest hospital (by road): {hosp['network_distance_km']} km<br><br>"
        f"<b>Dynamic total: {dyn['total_response_min']} min</b><br>"
        f"Static total: {stat['total_response_min']} min<br>"
        f"<span style='color:green'>Saved: {cmp['time_saved_min']} min "
        f"({cmp['time_improvement_pct']}%)</span>"
    )
).add_to(lg["p1_markers"])

# Dispatched ambulance
folium.Marker(
    [amb["lat"],amb["lon"]],
    icon=folium.DivIcon(
        html=('<div style="background:#d62828;color:white;padding:3px 8px;'
              'border-radius:10px;font-size:11px;font-weight:bold;'
              'white-space:nowrap;border:2px solid white;'
              'box-shadow:0 0 5px rgba(0,0,0,0.5)">🚑 DISPATCHED</div>'),
        icon_size=(110,24),icon_anchor=(55,12)
    ),
    tooltip=folium.Tooltip(
        f"<b>🚑 Dispatched Ambulance</b><br>"
        f"{amb['label']}<br>"
        f"Straight-line to accident: {amb['straight_line_km']} km"
    )
).add_to(lg["p1_markers"])

# Selected hospital
folium.Marker(
    [hosp["lat"],hosp["lon"]],
    icon=folium.DivIcon(
        html=('<div style="background:#8e44ad;color:white;padding:3px 8px;'
              'border-radius:10px;font-size:11px;font-weight:bold;'
              'white-space:nowrap;border:2px solid white;'
              'box-shadow:0 0 5px rgba(0,0,0,0.5)">🏥 DESTINATION</div>'),
        icon_size=(130,24),icon_anchor=(65,12)
    ),
    tooltip=folium.Tooltip(
        f"<b>🏥 Destination Hospital</b><br>{hosp['name']}<br>"
        f"Network distance: {hosp['network_distance_km']} km"
    )
).add_to(lg["p1_markers"])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MULTI-SCENARIO ROUTES (morning rush + night, lighter weight)
# ══════════════════════════════════════════════════════════════════════════════

CASE_COLORS = {"High":"#e74c3c","Medium":"#f39c12","Low":"#f1c40f"}

for res in p2_results:
    risk  = res["risk_level"]
    sc    = res["scenario"]
    col   = CASE_COLORS.get(risk,"#888")
    dyn_r = res["dynamic"]
    sta_r = res["static"]
    c     = res["comparison"]

    if sc == "morning_rush":
        target_dyn  = lg["p2_rush_dyn"]
        target_stat = lg["p2_rush_stat"]
        w = 4; op_d = 0.75; op_s = 0.60
    elif sc == "night":
        target_dyn  = lg["p2_night_dyn"]
        target_stat = None
        w = 3; op_d = 0.65; op_s = 0.55
    else:
        continue   # midday not separately drawn (shown in P1 detail only)

    if dyn_r["to_accident"]["coords"]:
        folium.PolyLine(
            dyn_r["to_accident"]["coords"],
            color=col,weight=w,opacity=op_d,
            tooltip=folium.Tooltip(
                f"<b>[P2 {sc}] Dynamic dispatch — {risk} risk</b><br>"
                f"Time: {dyn_r['total_response_min']} min | "
                f"Saved: {c['time_saved_min']} min"
            )
        ).add_to(target_dyn)
        if dyn_r["to_hospital"]["coords"]:
            folium.PolyLine(
                dyn_r["to_hospital"]["coords"],
                color=col,weight=w,opacity=op_d,dash_array="3 3",
                tooltip=folium.Tooltip(
                    f"<b>[P2 {sc}] Dynamic transport — {risk} risk</b><br>"
                    f"To hospital: {dyn_r['to_hospital']['estimated_time_min']} min"
                )
            ).add_to(target_dyn)

    if target_stat and sta_r["to_accident"]["coords"]:
        folium.PolyLine(
            sta_r["to_accident"]["coords"],
            color="#666",weight=3,opacity=op_s,dash_array="8 4",
            tooltip=folium.Tooltip(
                f"<b>[P2 {sc}] Static dispatch — {risk} risk</b><br>"
                f"Time: {sta_r['total_response_min']} min"
            )
        ).add_to(target_stat)

# Add P2 accident markers
for case in accident_cases:
    acc=case["accident_loc"]; risk=case["risk_level"]
    col=CASE_COLORS.get(risk,"#888")
    folium.CircleMarker(
        acc,radius=12,color=col,fill=True,fill_color=col,
        fill_opacity=0.5,weight=2,
        tooltip=folium.Tooltip(
            f"<b>[P2] {RISK_LABELS[risk]} Accident</b><br>{case['zone_label']}")
    ).add_to(lg["p2_rush_dyn"])

# Add all layers to map
for layer in lg.values():
    layer.add_to(m)
folium.LayerControl(collapsed=False, position="topright").add_to(m)

# ══════════════════════════════════════════════════════════════════════════════
# PANELS
# ══════════════════════════════════════════════════════════════════════════════

# ── Left Panel — Phase 1 dispatch detail ──────────────────────────────────────
def _row(label, dyn_val, stat_val, highlight=False):
    fw = "bold" if highlight else "normal"
    dc = "#d62828" if highlight else "#333"
    sc = "#888"
    return (f'<tr style="border-bottom:1px solid #f0f0f0">'
            f'<td style="padding:3px 5px;font-size:11px;color:#555">{label}</td>'
            f'<td style="padding:3px 6px;text-align:right;color:{dc};font-weight:{fw}">{dyn_val}</td>'
            f'<td style="padding:3px 6px;text-align:right;color:{sc}">{stat_val}</td>'
            f'</tr>')

p1d=p1_result["dynamic"]; p1s=p1_result["static"]; p1c=p1_result["comparison"]
win_col="#d62828" if p1c["winner"]=="dynamic" else "#1a6faf"
win_lbl="🚀 DYNAMIC" if p1c["winner"]=="dynamic" else "📏 STATIC"
lf_note=(
    '<div style="color:#8e44ad;font-size:10px;margin-top:3px">'
    '⚡ Dynamic longer in km but faster in time</div>'
    if p1c["dynamic_longer_but_faster"] else ""
)

panel_p1 = f"""
<div style="position:fixed;top:18px;left:18px;z-index:1000;
  background:rgba(255,255,255,0.97);padding:14px 16px;border-radius:12px;
  border:1px solid #ddd;font-family:Arial,sans-serif;font-size:12px;
  min-width:320px;box-shadow:3px 3px 14px rgba(0,0,0,0.18)">

  <div style="font-size:14px;font-weight:bold;margin-bottom:2px">
    🚑 Thrissur Emergency Dispatch  v3.0
  </div>
  <div style="color:#888;font-size:10px;margin-bottom:10px">
    Phase 1 · High-risk · Morning Rush (08:00)
  </div>

  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f8f8;font-size:10px;color:#777">
      <th style="padding:3px 5px;text-align:left">Metric</th>
      <th style="padding:3px 6px;text-align:right;color:#d62828">Dynamic</th>
      <th style="padding:3px 6px;text-align:right">Static</th>
    </tr>
    {_row("Dispatch time (min)", p1d["to_accident"]["estimated_time_min"], p1s["to_accident"]["estimated_time_min"])}
    {_row("Dispatch distance (km)", p1d["to_accident"]["distance_km"], p1s["to_accident"]["distance_km"])}
    {_row("Dispatch congestion", p1d["to_accident"]["avg_congestion"], p1s["to_accident"]["avg_congestion"])}
    {_row("On-scene delay (min)", ON_SCENE_DELAY_MIN, ON_SCENE_DELAY_MIN)}
    {_row("Transport time (min)", p1d["to_hospital"]["estimated_time_min"], p1s["to_hospital"]["estimated_time_min"])}
    {_row("Transport distance (km)", p1d["to_hospital"]["distance_km"], p1s["to_hospital"]["distance_km"])}
    <tr style="background:#f8f8f8">
      <td style="padding:4px 5px;font-weight:bold">TOTAL (min)</td>
      <td style="padding:4px 6px;text-align:right;color:#d62828;font-weight:bold;font-size:13px">{p1d["total_response_min"]}</td>
      <td style="padding:4px 6px;text-align:right;font-size:13px">{p1s["total_response_min"]}</td>
    </tr>
  </table>

  <div style="margin-top:10px;padding:8px;background:#f0fff4;border-radius:6px;
    border-left:4px solid {win_col}">
    <b style="color:{win_col}">Winner: {win_lbl}</b><br>
    <span style="font-size:11px">
      Time saved: <b>{p1c["time_saved_min"]} min</b>
      ({p1c["time_improvement_pct"]}%)<br>
      Congestion reduced: {p1c["congestion_reduction"]:.3f}
    </span>
    {lf_note}
  </div>

  <div style="margin-top:10px;font-size:11px;color:#555">
    <b>Ambulance:</b> {p1_result["ambulance"]["label"]}<br>
    <b>Hospital:</b> {p1_result["hospital"]["name"]}
    ({p1_result["hospital"]["network_distance_km"]} km by road)
  </div>

  <div style="margin-top:8px;padding-top:6px;border-top:1px solid #eee;
    font-size:10px;color:#aaa">
    Toggle layers → panel top-right | P2 multi-scenario routes available
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(panel_p1))

# ── Right Panel — Phase 2 summary ─────────────────────────────────────────────
p2_rows_html = ""
for res in p2_results:
    if res["scenario"] != "morning_rush": continue
    dyn_r=res["dynamic"]; sta_r=res["static"]; c=res["comparison"]
    col=CASE_COLORS.get(res["risk_level"],"#888")
    win_b=(f'<span style="background:#d62828;color:white;padding:1px 5px;'
           f'border-radius:3px;font-size:9px">▲ Dyn</span>'
           if c["winner"]=="dynamic" else
           f'<span style="background:#888;color:white;padding:1px 5px;'
           f'border-radius:3px;font-size:9px">Stat</span>')
    p2_rows_html += (
        f'<tr style="border-bottom:1px solid #eee">'
        f'<td style="padding:3px 5px">'
        f'<span style="color:{col};font-weight:bold">{res["risk_level"]}</span></td>'
        f'<td style="padding:3px 5px;text-align:right;color:#d62828">'
        f'{dyn_r["total_response_min"]}</td>'
        f'<td style="padding:3px 5px;text-align:right">'
        f'{sta_r["total_response_min"]}</td>'
        f'<td style="padding:3px 5px;text-align:center">{win_b}</td>'
        f'</tr>'
    )

panel_p2 = f"""
<div style="position:fixed;bottom:70px;left:18px;z-index:1000;
  background:rgba(255,255,255,0.96);padding:12px 14px;border-radius:10px;
  border:1px solid #ddd;font-family:Arial,sans-serif;font-size:12px;
  min-width:260px;box-shadow:2px 2px 10px rgba(0,0,0,0.16)">
  <div style="font-weight:bold;margin-bottom:6px">
    📊 Phase 2 — Morning Rush Summary
  </div>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#f8f8f8;font-size:10px;color:#777">
      <th style="padding:2px 5px;text-align:left">Risk</th>
      <th style="padding:2px 5px;text-align:right;color:#d62828">Dyn (min)</th>
      <th style="padding:2px 5px;text-align:right">Sta (min)</th>
      <th style="padding:2px 5px;text-align:center">Win</th>
    </tr>
    {p2_rows_html}
  </table>
  <div style="margin-top:7px;font-size:10px;color:#888">
    Dynamic wins {n_wins}/{len(p2_results)} · Avg saved: {avg_saved:.1f} min
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(panel_p2))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_html = """
<div style="position:fixed;bottom:70px;right:18px;z-index:1000;
  background:rgba(255,255,255,0.96);padding:12px 16px;border-radius:10px;
  border:1px solid #ddd;font-family:Arial,sans-serif;font-size:12px;
  line-height:1.9;box-shadow:2px 2px 8px rgba(0,0,0,0.16)">
<b>🗺️ Legend</b><br>

<b style="font-size:10px;color:#777">PHASE 1 ROUTES</b><br>
<span style="color:#d62828;font-size:14px">━━</span> Dynamic: Amb → Accident<br>
<span style="color:#1a6faf;font-size:14px">╌╌</span> Static:  Hosp → Accident<br>
<span style="color:#8e44ad;font-size:14px">━━</span> Dynamic: Accident → Hosp<br>
<span style="color:#1f7a6e;font-size:14px">╌╌</span> Static:  Accident → Hosp<br>

<b style="font-size:10px;color:#777">CONGESTION HEAT</b><br>
<span style="color:#2dc653">●</span> Clear &lt;0.25 &nbsp;
<span style="color:#f6c90e">●</span> Moderate &lt;0.50<br>
<span style="color:#f77f00">●</span> Heavy &lt;0.75 &nbsp;
<span style="color:#d62828">●</span> Severe ≥0.75<br>

<b style="font-size:10px;color:#777">ACCIDENT RISK</b><br>
<span style="color:#d62828;font-size:16px">●</span> High &nbsp;
<span style="color:#f77f00;font-size:13px">●</span> Medium &nbsp;
<span style="color:#f6c90e;font-size:10px">●</span> Low<br>

<b style="font-size:10px;color:#777">MARKERS</b><br>
⚠️ Accident scene &nbsp; 🚑 Dispatched<br>
<span style="background:crimson;color:white;padding:0 5px;
  border-radius:50%;font-weight:bold">+</span> Hospital
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "day3_dispatch_output.html"
m.save(OUT)

print(SEP)
print(f"  ✅  Map saved → {OUT}")
print(SEP)
print(f"\n  Phase 1  : 1 accident × 1 scenario (morning rush)  → full narrative")
print(f"  Phase 2  : {len(accident_cases)} accidents × {len(SCENARIOS)} scenarios "
      f"= {len(p2_results)} dispatches")
print(f"  Dynamic wins : {n_wins}/{len(p2_results)}")
print(f"  Avg time saved : {avg_saved:.2f} min")
print(f"  Ambulances   : {len(vehicle_locations)}")
print(f"  Hospitals    : {len(hospital_locations)}")
print(f"  Incidents    : {len(df)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 PREP — Save clustering data for Flask app
# ══════════════════════════════════════════════════════════════════════════════
import pickle

clustering_data = {
    "kmeans":     kmeans,
    "scaler":     scaler_g,
    "ambulances": vehicle_locations,
    "hospitals":  hospital_locations,
}

PKL_OUT = "clustering_data.pkl"
with open(PKL_OUT, "wb") as _f:
    pickle.dump(clustering_data, _f)

print(SEP)
print(f"  ✅  clustering_data.pkl saved → {PKL_OUT}")
print(f"      ({len(vehicle_locations)} ambulances, {len(hospital_locations)} hospitals)")
print(f"      Ready for:  python app.py")
print(SEP)