/* ══════════════════════════════════════════════════════════════
   main.js  —  Emergency Routing v6.1 (full)
   Added: draggable panels, status panel, separate route layers,
          panel toggle, hospital dispatch origin display
   All original result logic and IDs preserved.
══════════════════════════════════════════════════════════════ */

'use strict';

// ══ Constants ═══════════════════════════════════════════════════════════════
const ZONE_NAMES = {
  0: 'Zone N (North)',
  1: 'Zone NE (North-East)',
  2: 'Zone E (East)',
  3: 'Zone C (Central)',
  4: 'Zone W (West)',
  5: 'Zone S (South)',
};

const BASE_DISPATCH  = 4200;   // ms — scaled by congestion
const BASE_TRANSPORT = 3600;   // ms
const DUR_ONSCENE    = 1800;   // ms

const CENTER_LAT       = 10.5276;
const CENTER_LON       = 76.2144;
const SERVICE_RADIUS_M = 7500;

const RUSH_HOURS  = new Set([7,8,9,10,16,17,18,19,20]);
const NIGHT_HOURS = new Set([23,0,1,2,3,4,5]);


// ══ Map ═══════════════════════════════════════════════════════════════════════
const map = L.map('map', { zoomControl: false, preferCanvas: false })
              .setView([CENTER_LAT, CENTER_LON], 13);

L.tileLayer(
  'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      + ' &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20,
  }
).addTo(map);

L.control.zoom({ position: 'topright' }).addTo(map);

const _miniOsm = L.tileLayer(
  'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  { attribution: '&copy; OpenStreetMap contributors' }
);
new L.Control.MiniMap(_miniOsm, {
  position: 'bottomright', width: 150, height: 150,
  collapsedWidth: 25, collapsedHeight: 25,
  zoomLevelOffset: -5, toggleDisplay: true,
}).addTo(map);


// ══ Layer groups — separate for each route type ═══════════════════════════════
const lyrAmb          = L.featureGroup().addTo(map);
const lyrHosp         = L.featureGroup().addTo(map);
const lyrBoundary     = L.featureGroup().addTo(map);
const lyrDynamicDisp  = L.featureGroup().addTo(map);
const lyrDynamicTrans = L.featureGroup().addTo(map);
const lyrStaticDisp   = L.featureGroup().addTo(map);
const lyrStaticTrans  = L.featureGroup().addTo(map);
const lyrCongestion   = L.featureGroup().addTo(map);

L.control.layers(null, {
  '🚑 Standby Ambulances':       lyrAmb,
  '🏥 Hospitals':                lyrHosp,
  '📍 Service Area':             lyrBoundary,
  '🔴 Dynamic: Amb → Accident':  lyrDynamicDisp,
  '🟣 Dynamic: Accident → Hosp': lyrDynamicTrans,
  '🔵 Static:  Hosp → Accident': lyrStaticDisp,
  '🩵 Static:  Accident → Hosp': lyrStaticTrans,
  '🌡️ Congestion Heat':          lyrCongestion,
}, { position: 'topright', collapsed: false }).addTo(map);


// ══ State ═══════════════════════════════════════════════════════════════════
let accMarker   = null;
let accRing     = null;
let movingAmb   = null;
let animRunning = false;
let history     = [];


// ══ UI helpers ══════════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);
function _show(id) { $(id) && $(id).classList.remove('hidden'); }
function _hide(id) { $(id) && $(id).classList.add('hidden'); }


// ══ Hour slider + congestion dot ════════════════════════════════════════════
const hourSlider   = document.getElementById('hourSlider');
const hourLabel    = document.getElementById('hourLabel');
const minuteSelect = document.getElementById('minuteSelect');

function updateHourLabel() {
  const h = parseInt(hourSlider.value);
  const m = minuteSelect.value.padStart(2, '0');
  hourLabel.textContent = `${String(h).padStart(2, '0')}:${m}`;
  const pct = (h / 23 * 100).toFixed(1);
  hourSlider.style.background = `linear-gradient(90deg, #d62828 ${pct}%, #ddd ${pct}%)`;
  const dot = $('congDot');
  if (!dot) return;
  if (RUSH_HOURS.has(h)) {
    dot.style.background = '#d62828';
    dot.title = 'Rush hour — heavy traffic (×1.7–1.8)';
  } else if (NIGHT_HOURS.has(h)) {
    dot.style.background = '#2dc653';
    dot.title = 'Night — clear roads (×0.4)';
  } else {
    dot.style.background = '#f6c90e';
    dot.title = 'Moderate traffic';
  }
}
hourSlider.addEventListener('input', updateHourLabel);
minuteSelect.addEventListener('change', updateHourLabel);
updateHourLabel();


// ══ Haversine (metres) ═══════════════════════════════════════════════════════
function _haversineM(lat1, lon1, lat2, lon2) {
  const R  = 6371000;
  const dL = (lat2 - lat1) * Math.PI / 180;
  const dP = (lon2 - lon1) * Math.PI / 180;
  const a  = Math.sin(dL/2)**2
           + Math.cos(lat1*Math.PI/180) * Math.cos(lat2*Math.PI/180) * Math.sin(dP/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}


// ══ Boundary circle ═════════════════════════════════════════════════════════
function _drawBoundary() {
  lyrBoundary.clearLayers();
  L.circle([CENTER_LAT, CENTER_LON], {
    radius: SERVICE_RADIUS_M,
    color: '#d62828', weight: 2.5, opacity: 0.70,
    fillColor: '#d62828', fillOpacity: 0.03,
    dashArray: '9 6',
  })
  .bindTooltip('📍 <b>7.5 km Service Boundary</b><br>Accident must be placed inside this area', { sticky: true })
  .addTo(lyrBoundary);
  L.circleMarker([CENTER_LAT, CENTER_LON], {
    radius: 5, color: '#d62828', fillColor: '#d62828', fillOpacity: 0.75, weight: 1.5,
  })
  .bindTooltip('<b>🏙️ Thrissur City Centre</b><br>Dispatch HQ', { sticky: true })
  .addTo(lyrBoundary);
  const labelLat = CENTER_LAT + SERVICE_RADIUS_M / 111320;
  L.marker([labelLat, CENTER_LON], {
    icon: L.divIcon({
      className: '',
      html: `<div style="background:rgba(214,40,40,0.88);color:#fff;padding:2px 10px;border-radius:9px;font-size:9px;font-weight:bold;white-space:nowrap;letter-spacing:0.07em;border:1.5px solid white;box-shadow:0 1px 5px rgba(0,0,0,0.20);font-family:Arial,sans-serif;">⭕ 7.5 km SERVICE AREA</div>`,
      iconSize: [150, 20], iconAnchor: [75, 10],
    }),
    interactive: false, zIndexOffset: -100,
  }).addTo(lyrBoundary);
}


// ══ Init ════════════════════════════════════════════════════════════════════
async function init() {
  _drawBoundary();
  try {
    const r = await fetch('/initial_data');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    _renderAmbulances(data.ambulances);
    _renderHospitals(data.hospitals);
    $('histSummary').textContent =
      `${data.ambulances.length} ambulances deployed · ${data.hospitals.length} hospitals loaded · No runs yet`;
  } catch (e) {
    console.error('Init error:', e);
    $('histSummary').textContent = 'Failed to load map data — check Flask is running';
  }
}

function _renderAmbulances(ambs) {
  lyrAmb.clearLayers();
  const C = {
    High:   ['#d62828', '#e55555'],
    Medium: ['#f77f00', '#ffa040'],
    Low:    ['#d4ac00', '#f6c90e'],
  };
  ambs.forEach(a => {
    const [stroke, fill] = C[a.risk] || C.Low;
    L.circleMarker([a.lat, a.lon], {
      radius: 5, color: stroke, fillColor: fill, fillOpacity: 0.88, weight: 1.5,
    })
    .bindTooltip(`<b>🚑 ${a.risk || 'Low'} Risk Ambulance</b><br>${a.label || ''}`, { sticky: true })
    .addTo(lyrAmb);
  });
}

function _renderHospitals(hosps) {
  lyrHosp.clearLayers();
  hosps.forEach(h => {
    L.marker([h.lat, h.lon], {
      icon: L.divIcon({
        className: '',
        html: `<div style="background:crimson;color:white;border-radius:50%;width:15px;height:15px;line-height:15px;text-align:center;font-size:10px;font-weight:bold;border:2px solid rgba(255,255,255,0.9);box-shadow:0 1px 5px rgba(0,0,0,0.25);font-family:Arial,sans-serif;">+</div>`,
        iconSize: [15, 15], iconAnchor: [7, 7],
      }),
    })
    .bindTooltip(`<b>🏥 ${h.name || 'Hospital'}</b>`, { sticky: true })
    .addTo(lyrHosp);
  });
}


// ══ Map click — with 7.5 km boundary enforcement ════════════════════════════
map.on('click', function (e) {
  if (animRunning) return;
  const dist = _haversineM(CENTER_LAT, CENTER_LON, e.latlng.lat, e.latlng.lng);
  if (dist > SERVICE_RADIUS_M) {
    _showToast('⛔ Outside 7.5 km service area — place accident inside the red boundary');
    setTimeout(_hideToast, 3000);
    return;
  }
  if (accRing)   { map.removeLayer(accRing);   accRing  = null; }
  if (accMarker) { map.removeLayer(accMarker); accMarker = null; }
  accRing = L.circleMarker(e.latlng, {
    radius: 22, color: '#d62828', fillColor: 'transparent',
    fillOpacity: 0, weight: 2.5, opacity: 0.5, dashArray: '4 3',
  }).addTo(map);
  accMarker = L.marker(e.latlng, {
    icon: L.divIcon({
      className: '',
      html: `<div style="background:#d62828;color:white;border-radius:50%;width:32px;height:32px;line-height:32px;text-align:center;font-size:17px;border:3px solid white;box-shadow:0 0 10px rgba(214,40,40,0.65);font-family:Arial,sans-serif;">⚠</div>`,
      iconSize: [32, 32], iconAnchor: [16, 16],
    }),
    zIndexOffset: 500,
  })
  .bindTooltip('⚠️ <b>Accident Location</b><br>Select hour and click Run Simulation',
    { direction: 'top', offset: [0, -18] })
  .addTo(map);
  $('locText').textContent = `${e.latlng.lat.toFixed(5)},  ${e.latlng.lng.toFixed(5)}`;
  $('locText').style.color = '#d62828';
  $('locBox').classList.add('placed');
  $('runBtn').disabled = false;
});


// ══ Run Simulation ══════════════════════════════════════════════════════════
async function runSimulation() {
  if (!accMarker || animRunning) return;
  const hour   = parseInt(hourSlider.value);
  const minute = parseInt(minuteSelect.value);
  const latlng = accMarker.getLatLng();
  _setStatus('active', 'ROUTING');
  _show('secLoading');
  _hide('secResults');
  $('loadStep').textContent = 'Analysing network (18,885 nodes)...';
  $('runBtn').disabled = true;
  let result;
  try {
    $('loadStep').textContent = 'Running dynamic routing on 44,866 edges...';
    const r = await fetch('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: latlng.lat, lon: latlng.lng, hour, minute }),
    });
    $('loadStep').textContent = 'Building route geometry...';
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.error || `HTTP ${r.status}`);
    }
    result = await r.json();
  } catch (err) {
    console.error(err);
    _hide('secLoading');
    _setStatus('standby', 'STANDBY');
    $('runBtn').disabled = false;
    alert(`Dispatch failed: ${err.message}`);
    return;
  }
  _hide('secLoading');
  _handleResult(result, `${hour}:${minute.toString().padStart(2, '0')}`);
}


// ══ Status panel functions ═══════════════════════════════════════════════════
let currentLegStartTime = null;
let currentLegDuration = 0;
let currentLegName = '';
let currentLegEstimate = 0;

function showStatusPanel(phase, estimateMin, legName) {
  const panel = document.getElementById('statusPanel');
  panel.classList.remove('hidden');
  document.getElementById('statusPhaseText').textContent = phase;
  document.getElementById('statusEstimate').textContent = estimateMin.toFixed(1);
  document.getElementById('statusLeg').textContent = legName;
  document.getElementById('statusProgress').style.width = '0%';
  document.getElementById('statusElapsed').textContent = '0.0';
}

function updateStatusPanel(now) {
  if (!currentLegStartTime) return;
  const elapsedMs = now - currentLegStartTime;
  const elapsedMin = elapsedMs / 60000;
  const progress = Math.min(elapsedMs / currentLegDuration, 1);
  document.getElementById('statusElapsed').textContent = elapsedMin.toFixed(1);
  document.getElementById('statusProgress').style.width = (progress * 100) + '%';
}

function hideStatusPanel() {
  document.getElementById('statusPanel').classList.add('hidden');
  currentLegStartTime = null;
}


// ══ Congestion-aware duration ════════════════════════════════════════════════
function _calcDuration(avgCong, base) {
  const c = (typeof avgCong === 'number') ? Math.min(1, Math.max(0, avgCong)) : 0.3;
  return base * (0.65 + c * 1.35);
}


// ══ easeInOutQuad ════════════════════════════════════════════════════════════
function _ease(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}


// ══ Animated leg — RAF + easing + siren flicker + status updates ════════════
function _animateLeg({ coords, color, weight, opacity, duration, tooltip, onComplete }) {
  if (!coords || coords.length < 2) { onComplete && onComplete(); return; }

  const isDispatch = tooltip.includes('Ambulance → Accident');
  currentLegName = isDispatch ? 'to accident' : 'to hospital';
  currentLegDuration = duration;
  currentLegStartTime = performance.now();
  const estimateMin = duration / 60000;
  showStatusPanel(isDispatch ? 'Dispatching' : 'Transporting', estimateMin, currentLegName);

  // Determine target layer based on leg type
  const targetLayer = isDispatch ? lyrDynamicDisp : lyrDynamicTrans;

  const shadow = L.polyline([], {
    color: color, weight: weight + 9, opacity: 0.14, lineCap: 'round',
  }).addTo(targetLayer);

  const trace = L.polyline([], {
    color, weight, opacity, lineCap: 'round', lineJoin: 'round',
  })
  .bindTooltip(tooltip || '', { sticky: true })
  .addTo(targetLayer);

  if (movingAmb) { targetLayer.removeLayer(movingAmb); }

  function _makeAmbIcon(bg) {
    return L.divIcon({
      className: '',
      html: `<div style="background:${bg};color:white;padding:3px 9px;border-radius:10px;font-size:13px;font-weight:bold;white-space:nowrap;border:2.5px solid white;box-shadow:0 0 12px ${bg === '#d62828' ? 'rgba(214,40,40,0.75)' : 'rgba(26,80,200,0.75)'};font-family:Arial,sans-serif;transition:all 0.2s;">🚑</div>`,
      iconSize: [46, 26], iconAnchor: [23, 13],
    });
  }

  movingAmb = L.marker(coords[0], {
    icon: _makeAmbIcon('#d62828'), zIndexOffset: 900,
  }).addTo(targetLayer);

  let sirenRed = true;
  const sirenTimer = setInterval(() => {
    if (!movingAmb) { clearInterval(sirenTimer); return; }
    sirenRed = !sirenRed;
    movingAmb.setIcon(_makeAmbIcon(sirenRed ? '#d62828' : '#1a50c8'));
  }, 340);

  const n         = coords.length;
  const startTime = performance.now();
  let   lastIdx   = 0;

  function frame(now) {
    const rawT   = Math.min((now - startTime) / duration, 1);
    const t      = _ease(rawT);
    const target = Math.min(Math.floor(t * n), n - 1);

    for (let i = lastIdx; i <= target; i++) {
      trace.addLatLng(coords[i]);
      shadow.addLatLng(coords[i]);
    }
    lastIdx = target + 1;

    if (movingAmb) movingAmb.setLatLng(coords[target]);

    updateStatusPanel(now);

    if (rawT < 1) {
      requestAnimationFrame(frame);
    } else {
      clearInterval(sirenTimer);
      trace.setLatLngs(coords);
      shadow.setLatLngs(coords);
      if (movingAmb) { targetLayer.removeLayer(movingAmb); movingAmb = null; }
      onComplete && onComplete();
    }
  }

  requestAnimationFrame(frame);
}


// ══ DivIcon badge helpers ════════════════════════════════════════════════════
function _addHospLabel(latlng, name, type) {
  const isDyn = type === 'dyn';
  const bg    = isDyn ? '#8e44ad' : '#1f7a6e';
  const label = isDyn ? 'DYN HOSPITAL' : 'STA HOSPITAL';
  const note  = isDyn ? 'Network-distance (Dijkstra)' : 'Haversine straight-line';
  L.marker(latlng, {
    icon: L.divIcon({
      className: '',
      html: `<div style="background:${bg};color:white;padding:3px 9px;border-radius:10px;font-size:11px;font-weight:bold;white-space:nowrap;border:2px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.28);font-family:Arial,sans-serif;">🏥 ${label}</div>`,
      iconSize: [135, 24], iconAnchor: [67, 12],
    }),
    zIndexOffset: 600,
  })
  .bindTooltip(`<b>🏥 ${name}</b><br><i>${note}</i>`, { sticky: true })
  .addTo(lyrHosp); // add to hospital layer
}

function _addDispatchedLabel(latlng, type, name) {
  const label = type === 'hospital' ? `🏥 HOSPITAL DISPATCH: ${name}` : '🚑 DISPATCHED';
  L.marker(latlng, {
    icon: L.divIcon({
      className: '',
      html: `<div style="background:#d62828;color:white;padding:3px 9px;border-radius:10px;font-size:11px;font-weight:bold;white-space:nowrap;border:2px solid white;box-shadow:0 2px 8px rgba(214,40,40,0.45);font-family:Arial,sans-serif;">${label}</div>`,
      iconSize: type === 'hospital' ? [180, 24] : [125, 24],
      iconAnchor: type === 'hospital' ? [90, 12] : [62, 12],
    }),
    zIndexOffset: 700,
  })
  .bindTooltip(type === 'hospital' ? `Dispatch origin: ${name} hospital` : 'Pre-deployed ambulance dispatch origin', { sticky: true })
  .addTo(lyrAmb); // add to ambulance layer
}


// ══ Result handler — now uses separate layers and passes origin type ═══════
function _handleResult(data, timeLabel) {
  animRunning = true;

  const dyn  = data.dynamic;
  const stat = data.static;

  const ambLoc      = [data.ambulance.lat,       data.ambulance.lon]; // origin location
  const hospDynLoc  = [data.hospital_dynamic.lat, data.hospital_dynamic.lon];
  const hospStaLoc  = [data.hospital_static.lat,  data.hospital_static.lon];

  const dispCoords   = dyn.to_accident.coords  || [];
  const transCoords  = dyn.to_hospital.coords  || [];
  const sDispCoords  = stat.to_accident.coords || [];
  const sTransCoords = stat.to_hospital.coords || [];
  const congSegs     = [
    ...(dyn.to_accident.cong_segs  || []),
    ...(dyn.to_hospital.cong_segs  || []),
  ];

  // Clear all route layers
  lyrDynamicDisp.clearLayers();
  lyrDynamicTrans.clearLayers();
  lyrStaticDisp.clearLayers();
  lyrStaticTrans.clearLayers();
  lyrCongestion.clearLayers();

  // ── STATIC routes ─────────────────────────────────────────────────────────
  if (sDispCoords.length > 1) {
    L.polyline(sDispCoords, { color: '#1a6faf', weight: 11, opacity: 0.10 }).addTo(lyrStaticDisp);
    L.polyline(sDispCoords, {
      color: '#1a6faf', weight: 4, opacity: 0.65, dashArray: '11 7',
    })
    .bindTooltip(`<b>🔵 STATIC: Hosp → Accident</b><br>${stat.to_accident.distance_km} km · ${stat.to_accident.estimated_time_min} min<br>Congestion: ${stat.to_accident.avg_congestion}`)
    .addTo(lyrStaticDisp);
  }
  if (sTransCoords.length > 1) {
    L.polyline(sTransCoords, { color: '#1f7a6e', weight: 11, opacity: 0.10 }).addTo(lyrStaticTrans);
    L.polyline(sTransCoords, {
      color: '#1f7a6e', weight: 4, opacity: 0.65, dashArray: '11 7',
    })
    .bindTooltip(`<b>🩵 STATIC: Accident → Hosp</b><br>To: ${data.hospital_static.name}<br>${stat.to_hospital.distance_km} km · ${stat.to_hospital.estimated_time_min} min`)
    .addTo(lyrStaticTrans);
  }

  // ── Congestion heat segments ──────────────────────────────────────────────
  congSegs.forEach(seg => {
    if (seg.coords && seg.coords.length >= 2) {
      L.polyline(seg.coords, { color: seg.color, weight: 11, opacity: 0.30 }).addTo(lyrCongestion);
    }
  });

  // ── Hospital labels ───────────────────────────────────────────────────────
  _addHospLabel(hospDynLoc, data.hospital_dynamic.name, 'dyn');
  const sameHosp =
    Math.abs(hospDynLoc[0] - hospStaLoc[0]) < 0.0001 &&
    Math.abs(hospDynLoc[1] - hospStaLoc[1]) < 0.0001;
  if (!sameHosp) { _addHospLabel(hospStaLoc, data.hospital_static.name, 'sta'); }

  // ── PHASE 1: Dispatch ─────────────────────────────────────────────────────
  _showToast('📡 DISPATCHING AMBULANCE...');
  _setStatus('active', 'DISPATCHING');

  const dispDur  = _calcDuration(dyn.to_accident.avg_congestion, BASE_DISPATCH);
  const transDur = _calcDuration(dyn.to_hospital.avg_congestion, BASE_TRANSPORT);

  _animateLeg({
    coords:  dispCoords,
    color:   '#e82020',
    weight:  8,
    opacity: 0.95,
    duration: dispDur,
    tooltip: `<b>🔴 DYNAMIC: Ambulance → Accident</b><br>${dyn.to_accident.distance_km} km · <b>${dyn.to_accident.estimated_time_min} min</b><br>Congestion: ${dyn.to_accident.avg_congestion}`,
    onComplete: () => {
      // Get origin type and name from dispatch_origin
      const origin = data.dispatch_origin || data.ambulance;
      _addDispatchedLabel(ambLoc, origin.type, origin.name || origin.label);
      _showSceneFlash();
      _showToast('🚨 ON SCENE — PATIENT LOADING (4 min)');

      setTimeout(() => {
        _showToast('🏥 TRANSPORTING TO HOSPITAL...');
        _setStatus('active', 'TRANSPORT');

        _animateLeg({
          coords:  transCoords,
          color:   '#8e44ad',
          weight:  8,
          opacity: 0.92,
          duration: transDur,
          tooltip: `<b>🟣 DYNAMIC: Accident → Hospital</b><br>To: ${data.hospital_dynamic.name}<br>${dyn.to_hospital.distance_km} km · <b>${dyn.to_hospital.estimated_time_min} min</b><br>Congestion: ${dyn.to_hospital.avg_congestion}`,
          onComplete: () => {
            _hideToast();
            _showResults(data, timeLabel);
            hideStatusPanel();
          },
        });
      }, DUR_ONSCENE);
    },
  });

  const allPts = [...dispCoords, ...transCoords, ...sDispCoords, ...sTransCoords];
  if (allPts.length > 1) {
    try {
      map.fitBounds(L.polyline(allPts).getBounds().pad(0.12),
        { animate: true, duration: 0.8 });
    } catch (_) {}
  }
}


// ══ Show results — updated to display origin type in ambLabel ═══════════════
function _showResults(data, timeLabel) {
  animRunning = false;
  _setStatus('done', 'COMPLETE');

  const dyn = data.dynamic;
  const sta = data.static;
  const cmp = data.comparison;

  // KPI dashboard
  $('kpiResponse').textContent  = dyn.total_response_min.toFixed(1) + ' min';
  $('kpiSaved').textContent     = cmp.time_saved_min.toFixed(1) + ' min';
  $('kpiCongestion').textContent= dyn.avg_congestion.toFixed(2);
  $('kpiDistance').textContent  = dyn.total_distance_km.toFixed(2) + ' km';

  const set = (id, val) => {
    const el = $(id);
    if (!el) return;
    el.textContent = val;
    el.classList.remove('num-rev');
    void el.offsetWidth;
    el.classList.add('num-rev');
  };

  // Dispatch leg
  set('m_dyn_d',     dyn.to_accident.estimated_time_min.toFixed(2));
  set('m_dyn_ddist', dyn.to_accident.distance_km.toFixed(3));
  set('m_dyn_dcong', dyn.to_accident.avg_congestion.toFixed(3));
  set('m_sta_d',     sta.to_accident.estimated_time_min.toFixed(2));
  set('m_sta_ddist', sta.to_accident.distance_km.toFixed(3));
  set('m_sta_dcong', sta.to_accident.avg_congestion.toFixed(3));

  // Transport leg
  set('m_dyn_t',     dyn.to_hospital.estimated_time_min.toFixed(2));
  set('m_dyn_tdist', dyn.to_hospital.distance_km.toFixed(3));
  set('m_sta_t',     sta.to_hospital.estimated_time_min.toFixed(2));
  set('m_sta_tdist', sta.to_hospital.distance_km.toFixed(3));

  set('m_dyn_total', dyn.total_response_min.toFixed(2));
  set('m_sta_total', sta.total_response_min.toFixed(2));

  const isDyn  = cmp.winner === 'dynamic';
  const winBox = $('winnerBox');
  winBox.style.borderLeftColor = isDyn ? '#d62828' : '#1a6faf';
  winBox.style.background      = isDyn ? '#f0fff4' : '#f4f8ff';
  $('winnerText').style.color  = isDyn ? '#d62828' : '#1a6faf';
  $('winnerText').textContent  = isDyn ? 'Winner: 🚀 DYNAMIC' : 'Winner: 📏 STATIC';
  set('wSaved', `${Math.abs(cmp.time_saved_min).toFixed(2)} min`);
  set('wPct',   Math.abs(cmp.time_improvement_pct).toFixed(1));
  set('wCong',  cmp.congestion_reduction.toFixed(3));
  $('wNote').classList[cmp.dynamic_longer_but_faster ? 'remove' : 'add']('hidden');

  // Update ambulance label with origin type
  const origin = data.dispatch_origin || data.ambulance;
  if (origin.type === 'hospital') {
    $('ambLabel').textContent = `🏥 Hospital Dispatch: ${origin.name || origin.label || 'Hospital'}`;
  } else {
    $('ambLabel').textContent = origin.label || '—';
  }

  $('hospDynName').textContent = data.hospital_dynamic.name || '—';
  $('hospDynDist').textContent = data.hospital_dynamic.network_distance_km;
  $('hospStaName').textContent = data.hospital_static.name  || '—';
  $('hospStaDist').textContent = data.hospital_static.haversine_distance_km;

  $('zonePill').textContent    = ZONE_NAMES[data.zone_id] || `Zone ${data.zone_id}`;
  $('computeNote').textContent = data.compute_time_s != null
    ? `computed in ${data.compute_time_s}s` : '';

  _show('secResults');
  $('runBtn').disabled = false;

  // Insight
  _generateInsight(data);

  // History
  history.unshift({ timeLabel, dyn, sta, cmp });
  _renderHistory();
  _updateHistoryChart();
}


// ══ Insight generator ═══════════════════════════════════════════════════════
function _generateInsight(data) {
  const cmp = data.comparison;
  const dyn = data.dynamic;
  const h   = parseInt(hourSlider.value);
  const parts = [];
  if (cmp.winner === 'dynamic') {
    parts.push(`AI routing saved <b>${Math.abs(cmp.time_saved_min).toFixed(1)} min</b> (${Math.abs(cmp.time_improvement_pct).toFixed(0)}% faster) by dynamically avoiding congested roads.`);
  } else {
    parts.push(`Static baseline won — pre-positioned hospital proximity outweighed routing optimisation.`);
  }
  if (dyn.avg_congestion > 0.65) {
    parts.push(`High congestion (${(dyn.avg_congestion*100).toFixed(0)}%) — AI rerouted around blocked corridors.`);
  }
  if (cmp.dynamic_longer_but_faster) {
    parts.push(`Dynamic took a longer path in km but arrived faster — a clear detour beats a congested shortcut.`);
  }
  if (RUSH_HOURS.has(h)) {
    parts.push(`Rush-hour active at ${h}:00 — congestion ×1.7–1.8.`);
  } else if (NIGHT_HOURS.has(h)) {
    parts.push(`Night scenario at ${h}:00 — clear roads (×0.4).`);
  }
  const box  = $('insightBox');
  const txt  = $('insightText');
  if (txt) txt.innerHTML = parts.join(' ');
  if (box) box.classList.remove('hidden');
}


// ══ History panel (unchanged) ════════════════════════════════════════════════
function _renderHistory() {
  const tbody = $('histBody');
  const sumEl = $('histSummary');
  if (history.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="padding:8px 5px;text-align:center;color:#bbb;font-size:11px;font-style:italic;">Run a simulation to see history</td></tr>`;
    sumEl.textContent = 'No runs yet';
    return;
  }
  const dynWins  = history.filter(r => r.cmp.winner === 'dynamic').length;
  const avgSaved = (history.reduce((a,r) => a + r.cmp.time_saved_min, 0) / history.length).toFixed(1);
  const bestSave = Math.max(...history.map(r => r.cmp.time_saved_min)).toFixed(1);
  const wr = $('statWinRate');  const as = $('statAvgSaved');  const bs = $('statBestSaved');
  if (wr) wr.textContent = (dynWins / history.length * 100).toFixed(0) + '%';
  if (as) as.textContent = avgSaved + ' min';
  if (bs) bs.textContent = bestSave + ' min';
  tbody.innerHTML = history.slice(0, 6).map(r => {
    const isDyn = r.cmp.winner === 'dynamic';
    const badge = isDyn
      ? `<span style="background:#d62828;color:white;padding:1px 5px;border-radius:3px;font-size:9px;">▲ Dyn</span>`
      : `<span style="background:#888;color:white;padding:1px 5px;border-radius:3px;font-size:9px;">Stat</span>`;
    return `
      <tr style="border-bottom:1px solid #eee;">
        <td style="padding:3px 5px;font-size:11px;color:#555;">${r.timeLabel}</td>
        <td style="padding:3px 5px;text-align:right;color:#d62828;">${r.dyn.total_response_min.toFixed(2)}</td>
        <td style="padding:3px 5px;text-align:right;color:#555;">${r.sta.total_response_min.toFixed(2)}</td>
        <td style="padding:3px 5px;text-align:center;">${badge}</td>
      </tr>`;
  }).join('');
  sumEl.textContent = `Dynamic wins ${dynWins}/${history.length} · Avg saved: ${avgSaved} min`;
}

function _updateHistoryChart() {
  if (!window.historyChart) return;
  const labels  = history.slice(0, 10).map(r => r.timeLabel).reverse();
  const dynData = history.slice(0, 10).map(r => r.dyn.total_response_min).reverse();
  const staData = history.slice(0, 10).map(r => r.sta.total_response_min).reverse();
  historyChart.data.labels           = labels;
  historyChart.data.datasets[0].data = dynData;
  historyChart.data.datasets[1].data = staData;
  historyChart.update();
}


// ══ Clear all — added hideStatusPanel ═══════════════════════════════════════
function clearAll() {
  if (animRunning) return;
  if (accRing)   { map.removeLayer(accRing);   accRing  = null; }
  if (accMarker) { map.removeLayer(accMarker); accMarker = null; }
  lyrDynamicDisp.clearLayers();
  lyrDynamicTrans.clearLayers();
  lyrStaticDisp.clearLayers();
  lyrStaticTrans.clearLayers();
  lyrCongestion.clearLayers();
  movingAmb = null;
  _hideToast();
  $('locText').textContent = 'Click on map to place accident';
  $('locText').style.color = '#888';
  $('locBox').classList.remove('placed');
  $('runBtn').disabled = true;
  _hide('secLoading');
  _hide('secTimer');   // safe even if removed
  _hide('secResults');
  _setStatus('standby', 'STANDBY');
  $('kpiResponse').textContent  = '--';
  $('kpiSaved').textContent     = '--';
  $('kpiCongestion').textContent= '--';
  $('kpiDistance').textContent  = '--';
  const box = $('insightBox');
  if (box) box.classList.add('hidden');
  const fl = document.getElementById('sceneFlash');
  if (fl) fl.remove();
  hideStatusPanel();
}


// ══ On-scene flash ═══════════════════════════════════════════════════════════
function _showSceneFlash() {
  const ex = document.getElementById('sceneFlash');
  if (ex) ex.remove();
  const el = document.createElement('div');
  el.id = 'sceneFlash';
  document.body.appendChild(el);
  setTimeout(() => { if (el.parentNode) el.remove(); }, 2400);
}


// ══ Phase toast ═════════════════════════════════════════════════════════════
function _showToast(text) {
  const el = $('phaseToast');
  el.textContent = text;
  el.classList.remove('hidden');
}
function _hideToast() { $('phaseToast').classList.add('hidden'); }


// ══ Status badge ═══════════════════════════════════════════════════════════
function _setStatus(mode, label) {
  const badge = $('statusBadge');
  const dot   = $('statusDot');
  $('statusLabel').textContent = label;
  badge.className = `s-badge ${mode}`;
  dot.style.animationDuration = mode === 'active' ? '0.65s' : '1.6s';
  dot.style.animation = mode === 'done' ? 'none' : null;
}


// ══ Panel toggle & draggable ════════════════════════════════════════════════
function initPanelControls() {
  const panels = document.querySelectorAll('.draggable-panel');
  const toggleBtns = document.querySelectorAll('.toggle-btn');

  // Individual toggle
  toggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const panelId = btn.dataset.panel;
      const panel = document.getElementById(panelId);
      if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.classList.add('active');
      } else {
        panel.style.display = 'none';
        btn.classList.remove('active');
      }
    });
    // Set initial active state (all panels visible)
    btn.classList.add('active');
  });

  // Close buttons
  panels.forEach(panel => {
    const closeBtn = panel.querySelector('.panel-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.style.display = 'none';
        // Update toggle button
        const panelId = panel.id;
        toggleBtns.forEach(btn => {
          if (btn.dataset.panel === panelId) btn.classList.remove('active');
        });
      });
    }
  });

  // Dragging
  panels.forEach(panel => {
    const header = panel.querySelector('.panel-header');
    let isDragging = false;
    let offsetX, offsetY;

    header.addEventListener('mousedown', (e) => {
      if (e.target.closest('button')) return;
      e.preventDefault();
      const rect = panel.getBoundingClientRect();
      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;
      isDragging = true;
      panel.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      e.preventDefault();
      let left = e.clientX - offsetX;
      let top = e.clientY - offsetY;
      left = Math.max(0, Math.min(left, window.innerWidth - panel.offsetWidth));
      top = Math.max(20, Math.min(top, window.innerHeight - panel.offsetHeight));
      panel.style.left = left + 'px';
      panel.style.top = top + 'px';
      panel.style.bottom = 'auto';
      panel.style.right = 'auto';
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
      panel.style.cursor = '';
    });
  });
}

document.addEventListener('DOMContentLoaded', initPanelControls);


// ══ Boot ════════════════════════════════════════════════════════════════════
init();