import osmnx as ox
import random
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap, HeatMap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

print("Loading Thrissur road network...")

center_point = (10.5276, 76.2144)
G = ox.graph_from_point(center_point, dist=7500, network_type="drive")
nodes = list(G.nodes(data=True))
edges = list(G.edges(data=True))

print(f"Nodes: {len(nodes)} | Edges: {len(edges)}")

# -------------------------------------------------------
# STEP 1 — Realistic Accident Generation (1000 points)
# Accident probability weighted by:
#   - Road type (junctions > highways > residential)
#   - Known high-risk corridors in Thrissur
#   - Time-of-day clustering (simulated density spots)
# -------------------------------------------------------

print("Generating 1000 realistic accident points...")

# High-risk real-world hotspot seeds in Thrissur
# (Shoranur Rd, Palakkad Rd, Poonkunnam junction, Round South, etc.)
HOTSPOT_SEEDS = [
    (10.5210, 76.2120),   # Thrissur Round South
    (10.5300, 76.2200),   # Shoranur Road corridor
    (10.5350, 76.2050),   # Palakkad Road
    (10.5150, 76.2300),   # Irinjalakuda junction area
    (10.5420, 76.2180),   # Ollur junction
    (10.5080, 76.2080),   # Poonkunnam
    (10.5270, 76.2400),   # Kodakara stretch
    (10.5480, 76.2320),   # Chalakudy road entry
    (10.5050, 76.2250),   # Ayyanthole junction
    (10.5600, 76.2100),   # Guruvayur road
]

random.seed(42)
np.random.seed(42)

accident_data = []

# 60% of accidents: near known hotspots (Gaussian cluster around seed)
for _ in range(600):
    seed = random.choice(HOTSPOT_SEEDS)
    lat = seed[0] + np.random.normal(0, 0.008)
    lon = seed[1] + np.random.normal(0, 0.008)
    accident_data.append([lat, lon])

# 30% of accidents: on actual road network edges (junctions)
junction_nodes = [
    n for n in nodes
    if G.degree(n[0]) >= 3  # intersections only
]
for _ in range(300):
    node = random.choice(junction_nodes)
    lat = node[1]['y'] + np.random.normal(0, 0.001)
    lon = node[1]['x'] + np.random.normal(0, 0.001)
    accident_data.append([lat, lon])

# 10%: random road network (minor incidents anywhere)
for _ in range(100):
    node = random.choice(nodes)
    accident_data.append([node[1]['y'], node[1]['x']])

df = pd.DataFrame(accident_data, columns=["lat", "lon"])

# Clip to Thrissur bounding box
df = df[
    (df["lat"].between(10.48, 10.58)) &
    (df["lon"].between(76.17, 76.27))
].reset_index(drop=True)

print(f"Accidents after clipping: {len(df)}")

# -------------------------------------------------------
# STEP 2 — KMeans Zoning (6 zones, geographically scaled)
# -------------------------------------------------------

print("Applying KMeans 6-zone clustering...")

scaler_global = StandardScaler()
coords_scaled = scaler_global.fit_transform(df[["lat", "lon"]])

kmeans = KMeans(n_clusters=6, random_state=42, n_init=15, max_iter=500)
df["zone"] = kmeans.fit_predict(coords_scaled).astype(int)

# Soft, light fill colors for zones (pastel — background context only)
ZONE_COLORS = {
    0: "#b5d5f5",   # Ice blue
    1: "#fde8a8",   # Pale gold
    2: "#c3e6cb",   # Mint green
    3: "#f5c6cb",   # Blush pink
    4: "#d4b8e0",   # Lavender
    5: "#b2e0d9",   # Pale teal
}
ZONE_LABELS = {
    0: "Zone N (North)",
    1: "Zone NE (North-East)",
    2: "Zone E (East)",
    3: "Zone C (Central)",
    4: "Zone W (West)",
    5: "Zone S (South)",
}

# -------------------------------------------------------
# STEP 3 — DBSCAN per zone → risk classification
# -------------------------------------------------------

print("Running DBSCAN per zone...")

df["risk"] = "Low"
df["db_cluster"] = -1

# ── Strategy: percentile-based classification (robust to any eps) ────────────
# Fixed thresholds fail because cluster sizes depend on data density.
# Instead we rank clusters by size within each zone and assign risk by
# percentile — guaranteeing a realistic split regardless of absolute sizes:
#
#   Top    20% of clusters (by point count) → High   risk
#   Next   30% of clusters                 → Medium risk
#   Bottom 50% of clusters + all noise     → Low    risk
#
# DBSCAN eps=0.22 in standardised space gives 10-30 clusters per zone,
# each representing a geographically tight hotspot (≈200-400 m radius).
# min_samples=4 ensures at least 4 co-located accidents before a cluster forms.
# ─────────────────────────────────────────────────────────────────────────────
DBSCAN_EPS         = 0.22
DBSCAN_MIN_SAMPLES = 4

# Target proportion of ACCIDENTS (not clusters) in each risk band
# Noise points always go to Low; cluster points are distributed by rank
HIGH_PCT   = 0.20   # top 20% of clusters → High
MEDIUM_PCT = 0.30   # next 30% of clusters → Medium

cluster_centroids = []   # {zone, cluster_id, risk, lat, lon, size}

for zone_id in sorted(df["zone"].unique()):
    mask     = df["zone"] == zone_id
    zone_pts = df.loc[mask, ["lat", "lon"]].copy()

    if len(zone_pts) < DBSCAN_MIN_SAMPLES:
        continue

    scaled = StandardScaler().fit_transform(zone_pts)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(scaled)
    df.loc[mask, "db_cluster"] = labels

    # Collect all real clusters (exclude noise = -1) with their sizes
    zone_clusters = []
    for cid in set(labels):
        if cid == -1:
            continue
        cmask = mask & (df["db_cluster"] == cid)
        zone_clusters.append({
            "cid":  cid,
            "mask": cmask,
            "sz":   int(cmask.sum()),
        })

    if not zone_clusters:
        continue   # all noise in this zone → everything stays Low

    # Sort clusters by size descending — largest = most dangerous hotspot
    zone_clusters.sort(key=lambda x: x["sz"], reverse=True)
    n_clusters = len(zone_clusters)

    # Percentile cut-points (minimum 1 cluster per band when possible)
    n_high   = max(1, round(n_clusters * HIGH_PCT))
    n_medium = max(1, round(n_clusters * MEDIUM_PCT))

    # Assign risk by rank
    for rank, cl in enumerate(zone_clusters):
        if rank < n_high:
            risk = "High"
        elif rank < n_high + n_medium:
            risk = "Medium"
        else:
            risk = "Low"

        df.loc[cl["mask"], "risk"] = risk

        c_lat = df.loc[cl["mask"], "lat"].mean()
        c_lon = df.loc[cl["mask"], "lon"].mean()
        cluster_centroids.append({
            "zone":       zone_id,
            "cluster_id": cl["cid"],
            "risk":       risk,
            "lat":        c_lat,
            "lon":        c_lon,
            "size":       cl["sz"],
            "rank":       rank,
        })

print("\nRisk distribution:")
print(df["risk"].value_counts().to_string())
print(f"\nDBSCAN clusters found: {len(cluster_centroids)}")
for r in ["High", "Medium", "Low"]:
    cls = [c for c in cluster_centroids if c["risk"] == r]
    pts = sum(c["size"] for c in cls)
    print(f"  {r:6s}: {len(cls):2d} clusters  |  {pts:3d} accident points")
print()

# -------------------------------------------------------
# STEP 4 — Ambulance Placement  (per DBSCAN cluster, not per zone)
# Counts: High → 5 per cluster, Medium → 3, Low → 1
# Spread: ambulances placed in arc ~300 m around cluster centroid
# -------------------------------------------------------

print("Allocating ambulances...")

# Ambulances per HIGH/MEDIUM cluster; one standby per LOW cluster
AMBU_COUNT = {"High": 5, "Medium": 3, "Low": 1}
SPREAD     = 0.003   # ~300 m in degrees
rng        = np.random.default_rng(7)

vehicle_locations = []

for cc in cluster_centroids:
    ambu_n    = AMBU_COUNT[cc["risk"]]
    clat, clon = cc["lat"], cc["lon"]

    for i in range(ambu_n):
        # Evenly distribute in a circle around cluster centroid
        angle = (2 * np.pi * i) / max(ambu_n, 1)
        offset_lat = SPREAD * np.sin(angle) * rng.uniform(0.6, 1.1)
        offset_lon = SPREAD * np.cos(angle) * rng.uniform(0.6, 1.1)
        vehicle_locations.append({
            "lat":        clat + offset_lat,
            "lon":        clon + offset_lon,
            "zone":       cc["zone"],
            "cluster_id": cc["cluster_id"],
            "risk":       cc["risk"],
            "label":      (
                f"Ambulance | {ZONE_LABELS[cc['zone']]} | "
                f"{cc['risk']} cluster (size {cc['size']})"
            ),
        })

print(f"Total ambulances deployed: {len(vehicle_locations)}")
by_risk = {"High": 0, "Medium": 0, "Low": 0}
for v in vehicle_locations:
    by_risk[v["risk"]] += 1
for r, n in by_risk.items():
    print(f"  {r:6s}: {n} ambulances")

# -------------------------------------------------------
# STEP 5 — Fetch Hospitals (centroid-safe)
# -------------------------------------------------------

print("Fetching hospitals...")

tags = {"amenity": "hospital"}
hospitals_gdf = ox.features_from_point(center_point, tags=tags, dist=7500)

hospital_locations = []
for _, row in hospitals_gdf.iterrows():
    geom = row.geometry
    if geom is None:
        continue
    name = row.get("name", "Hospital")
    hospital_locations.append({
        "lat": geom.centroid.y,
        "lon": geom.centroid.x,
        "name": name if isinstance(name, str) else "Hospital"
    })

print(f"Hospitals found: {len(hospital_locations)}")

# -------------------------------------------------------
# STEP 6 — Color logic: dark/vivid per risk level
# Low → yellow (#f6c90e), Medium → orange (#f77f00), High → red (#d62828)
# Zone shown as filled convex hull polygon (clearly visible background)
# -------------------------------------------------------

from scipy.spatial import ConvexHull

RISK_FILL   = {"Low": "#f6c90e", "Medium": "#f77f00", "High": "#d62828"}
RISK_RADIUS = {"Low": 5, "Medium": 7, "High": 10}
RISK_OPACITY = {"Low": 0.55, "Medium": 0.75, "High": 0.92}
RISK_WEIGHT  = {"Low": 0.5, "Medium": 1.0, "High": 1.5}

# Solid border colors (darker version of each pastel zone)
ZONE_BORDER_COLORS = {
    0: "#2a7abf",   # Deeper blue
    1: "#c87f00",   # Deeper gold
    2: "#2e7d4f",   # Deeper green
    3: "#b5434a",   # Deeper pink/red
    4: "#7b4fa6",   # Deeper lavender
    5: "#1f7a6e",   # Deeper teal
}

# -------------------------------------------------------
# STEP 7 — Build Map
# -------------------------------------------------------

print("Building map...")

m = folium.Map(
    location=center_point,
    zoom_start=13,
    tiles="CartoDB positron"
)
MiniMap(toggle_display=True, position="bottomright").add_to(m)

# --- Layer groups for toggle control ---
layer_groups = {
    "Zones": folium.FeatureGroup(name="🗺️ Zone Boundaries", show=True),
    "High Risk": folium.FeatureGroup(name="🔴 High Risk Accidents", show=True),
    "Medium Risk": folium.FeatureGroup(name="🟠 Medium Risk Accidents", show=True),
    "Low Risk": folium.FeatureGroup(name="🟡 Low Risk Accidents", show=True),
    "Ambulances": folium.FeatureGroup(name="🚑 Ambulances", show=True),
    "Hospitals": folium.FeatureGroup(name="🏥 Hospitals", show=True),
    "Heatmap": folium.FeatureGroup(name="🌡️ Accident Heatmap", show=False),
}

# --- ZONE CONVEX HULL POLYGONS (drawn first = behind everything) ---
print("Drawing zone polygons...")

for zone_id in sorted(df["zone"].unique()):
    zone_pts = df[df["zone"] == zone_id][["lat", "lon"]].values

    if len(zone_pts) < 4:
        continue

    try:
        hull = ConvexHull(zone_pts)
        hull_coords = zone_pts[hull.vertices].tolist()
        # Close the polygon
        hull_coords.append(hull_coords[0])

        # Convert to [lat, lon] list for folium
        polygon_latlons = [[pt[0], pt[1]] for pt in hull_coords]

        fill_hex = ZONE_COLORS[zone_id]
        border_hex = ZONE_BORDER_COLORS[zone_id]
        label = ZONE_LABELS[zone_id]

        folium.Polygon(
            locations=polygon_latlons,
            color=border_hex,
            weight=2.5,
            fill=True,
            fill_color=fill_hex,
            fill_opacity=0.22,        # Clearly visible but not overwhelming
            dash_array="6 4",         # Dashed border distinguishes zones cleanly
            tooltip=folium.Tooltip(
                f"<b>{label}</b><br>"
                f"Accidents: {len(zone_pts)}<br>"
                f"High: {len(df[(df['zone']==zone_id) & (df['risk']=='High')])}&nbsp;"
                f"Medium: {len(df[(df['zone']==zone_id) & (df['risk']=='Medium')])}&nbsp;"
                f"Low: {len(df[(df['zone']==zone_id) & (df['risk']=='Low')])}"
            )
        ).add_to(layer_groups["Zones"])

        # Zone label marker at centroid
        clat, clon = zone_pts[:, 0].mean(), zone_pts[:, 1].mean()
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(
                html=(
                    f'<div style="'
                    f'background:{fill_hex};'
                    f'border:2px solid {border_hex};'
                    f'color:{border_hex};'
                    f'font-size:11px;font-weight:bold;'
                    f'padding:3px 7px;border-radius:12px;'
                    f'white-space:nowrap;'
                    f'box-shadow:0 1px 4px rgba(0,0,0,0.3);'
                    f'">{label}</div>'
                ),
                icon_size=(130, 26),
                icon_anchor=(65, 13)
            )
        ).add_to(layer_groups["Zones"])

    except Exception as e:
        print(f"  Zone {zone_id} hull failed: {e}")

# --- Heatmap layer (background density) ---
heat_data = [[row["lat"], row["lon"]] for _, row in df.iterrows()]
HeatMap(
    heat_data,
    radius=14,
    blur=18,
    max_zoom=14,
    gradient={"0.3": "blue", "0.55": "lime", "0.75": "orange", "1.0": "red"},
    name="Heatmap"
).add_to(layer_groups["Heatmap"])

# --- Accident points: vivid risk fill, thin neutral border ---
for _, row in df.iterrows():
    fill = RISK_FILL[row["risk"]]
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=RISK_RADIUS[row["risk"]],
        color="#555555",            # neutral dark grey border — zone shown by polygon
        fill=True,
        fill_color=fill,
        fill_opacity=RISK_OPACITY[row["risk"]],
        weight=0.6,
        tooltip=folium.Tooltip(
            f"<b>{ZONE_LABELS[row['zone']]}</b><br>"
            f"Risk: <b style='color:{fill}'>{row['risk']}</b><br>"
            f"Cluster ID: {int(row['db_cluster'])}"
        )
    ).add_to(layer_groups[f"{row['risk']} Risk"])

# --- Ambulances: distinct icon per risk level ---
AMBU_ICON_COLOR = {"High": "red", "Medium": "orange", "Low": "beige"}

for v in vehicle_locations:
    folium.Marker(
        location=[v["lat"], v["lon"]],
        icon=folium.Icon(
            color=AMBU_ICON_COLOR[v["risk"]],
            icon="ambulance",
            prefix="fa"
        ),
        tooltip=folium.Tooltip(
            f"<b>🚑 Ambulance</b><br>{v['label']}"
        )
    ).add_to(layer_groups["Ambulances"])

# --- Hospitals: small red cross with name tooltip ---
for h in hospital_locations:
    folium.Marker(
        location=[h["lat"], h["lon"]],
        icon=folium.DivIcon(
            html=(
                '<div style="'
                'font-size:14px;'
                'color:white;'
                'background:crimson;'
                'border-radius:50%;'
                'width:20px;height:20px;'
                'display:flex;align-items:center;justify-content:center;'
                'border:2px solid white;'
                'box-shadow:0 0 4px rgba(0,0,0,0.4);'
                'font-weight:bold;'
                '">+</div>'
            ),
            icon_size=(22, 22),
            icon_anchor=(11, 11)
        ),
        tooltip=folium.Tooltip(f"🏥 <b>{h['name']}</b>")
    ).add_to(layer_groups["Hospitals"])

# Add all layers to map
for lg in layer_groups.values():
    lg.add_to(m)

folium.LayerControl(collapsed=False, position="topright").add_to(m)

# -------------------------------------------------------
# STEP 8 — Stats Panel (top-left)
# -------------------------------------------------------

risk_counts = df["risk"].value_counts()
zone_counts = df["zone"].value_counts().sort_index()

zone_stats_rows = "".join([
    f'<tr><td style="padding:2px 6px">'
    f'<span style="background:{ZONE_COLORS[z]};padding:1px 6px;border-radius:3px;'
    f'border:1px solid #aaa">{ZONE_LABELS[z]}</span></td>'
    f'<td style="padding:2px 8px;text-align:center">{cnt}</td></tr>'
    for z, cnt in zone_counts.items()
])

stats_html = f"""
<div style="
    position: fixed;
    top: 20px; left: 20px;
    z-index: 1000;
    background: rgba(255,255,255,0.96);
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid #ccc;
    font-family: 'Arial', sans-serif;
    font-size: 12px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    min-width: 220px;
">
<div style="font-size:14px;font-weight:bold;margin-bottom:8px">
    📍 Thrissur Accident Risk Map
</div>
<div style="color:#555;margin-bottom:10px;font-size:11px">
    1000 simulated accidents · 6 KMeans zones · DBSCAN risk clusters
</div>

<b>Risk Summary</b>
<table style="width:100%;margin:4px 0 10px 0;border-collapse:collapse">
  <tr>
    <td><span style="color:#d62828;font-size:16px">●</span> High Risk</td>
    <td style="text-align:right"><b>{risk_counts.get("High", 0)}</b> incidents</td>
    <td style="text-align:right;color:#888">{int(len(vehicle_locations)/3 * 5 // 5)} ambu.</td>
  </tr>
  <tr>
    <td><span style="color:#f77f00;font-size:16px">●</span> Medium Risk</td>
    <td style="text-align:right"><b>{risk_counts.get("Medium", 0)}</b> incidents</td>
    <td style="text-align:right;color:#888"></td>
  </tr>
  <tr>
    <td><span style="color:#f6c90e;font-size:16px">●</span> Low Risk</td>
    <td style="text-align:right"><b>{risk_counts.get("Low", 0)}</b> incidents</td>
    <td style="text-align:right;color:#888"></td>
  </tr>
</table>

<b>Zone Breakdown</b>
<table style="width:100%;margin-top:4px;border-collapse:collapse">
{zone_stats_rows}
</table>

<div style="margin-top:10px;padding-top:8px;border-top:1px solid #eee;font-size:11px;color:#555">
    🚑 {len(vehicle_locations)} ambulances deployed &nbsp;|&nbsp;
    🏥 {len(hospital_locations)} hospitals
</div>
<div style="margin-top:6px;font-size:10px;color:#aaa">
    Toggle layers → top right panel
</div>
</div>
"""

m.get_root().html.add_child(folium.Element(stats_html))

# -------------------------------------------------------
# STEP 9 — Legend (bottom-right)
# -------------------------------------------------------

legend_html = """
<div style="
    position: fixed;
    bottom: 60px; right: 20px;
    z-index: 1000;
    background: rgba(255,255,255,0.95);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #ccc;
    font-family: Arial, sans-serif;
    font-size: 12px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    line-height: 2;
">
<b>🗺️ Legend</b><br>

<b style="font-size:11px;color:#777">ACCIDENT RISK (fill colour)</b><br>
<span style="color:#d62828;font-size:18px">●</span> High Risk &nbsp;
<span style="color:#f77f00;font-size:15px">●</span> Medium Risk &nbsp;
<span style="color:#f6c90e;font-size:12px">●</span> Low Risk<br>

<b style="font-size:11px;color:#777">ZONES (shaded polygon regions)</b><br>
<span style="background:#b5d5f5;border:2px solid #2a7abf;padding:0 8px;border-radius:3px">Zone N</span>&nbsp;
<span style="background:#fde8a8;border:2px solid #c87f00;padding:0 8px;border-radius:3px">Zone NE</span>&nbsp;
<span style="background:#c3e6cb;border:2px solid #2e7d4f;padding:0 8px;border-radius:3px">Zone E</span><br>
<span style="background:#f5c6cb;border:2px solid #b5434a;padding:0 8px;border-radius:3px">Zone C</span>&nbsp;
<span style="background:#d4b8e0;border:2px solid #7b4fa6;padding:0 8px;border-radius:3px">Zone W</span>&nbsp;
<span style="background:#b2e0d9;border:2px solid #1f7a6e;padding:0 8px;border-radius:3px">Zone S</span><br>

<b style="font-size:11px;color:#777">AMBULANCES (icon colour)</b><br>
🚑 Red = covers High &nbsp; Orange = Medium &nbsp; Beige = Low<br>

<b style="font-size:11px;color:#777">HOSPITALS</b><br>
<span style="background:crimson;color:white;padding:0 6px;border-radius:50%;font-weight:bold">+</span> Hospital (tap for name)
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

m.save("day1_output.html")
print("✅ Map saved → day1_output.html")
print(f"\nSummary:")
print(f"  Accidents   : {len(df)}")
print(f"  High risk   : {risk_counts.get('High', 0)}")
print(f"  Medium risk : {risk_counts.get('Medium', 0)}")
print(f"  Low risk    : {risk_counts.get('Low', 0)}")
print(f"  Ambulances  : {len(vehicle_locations)}")
print(f"  Hospitals   : {len(hospital_locations)}")