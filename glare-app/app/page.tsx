"use client";

import { toPng } from "html-to-image";
import { useEffect, useMemo, useRef, useState } from "react";

type TimeResult = {
  label: string;
  hour: number;
  sun_visible: boolean;
  altitude_deg: number | null;
  azimuth_deg: number | null;
  sunlit_area_sqft: number;
  coverage_ratio: number;
  max_penetration_ft: number;
};

type AnalyzeResponse = {
  ok: boolean;
  summary: {
    best_time_label: string | null;
    max_sunlit_area_sqft: number;
    average_coverage_ratio: number;
    max_penetration_ft: number;
  };
  times: TimeResult[];
  debug: {
    shading_count_generated: number;
  };
  error?: string;
};

type SavedScenario = {
  id: string;
  name: string;
  inputs: {
    roomWidth: number;
    roomDepth: number;
    roomHeight: number;
    windowWidth: number;
    windowHeight: number;
    sillHeight: number;
    windowOffset: number;
    latitude: number;
    orientation: "North" | "South" | "East" | "West";
    analysisDate: "03-21" | "06-21" | "12-21";
    timeMode: "full_day" | "9" | "12" | "15";
    hasShading: boolean;
    shadingType: "horizontal" | "vertical" | "eggcrate";
    horizontalCount: number;
    verticalCount: number;
    horizontalDepth: number;
    verticalDepth: number;
    shadingThickness: number;
    horizontalSpacing: number;
    verticalSpacing: number;
  };
  result: AnalyzeResponse;
};
type Pt2 = { x: number; y: number };

type SunPatchResult = {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  area: number;
  maxPenetration: number;
  floorPoints: Pt2[];
  hullPoints: Pt2[];
  altitudeDeg: number;
  azimuthDeg: number;
  segmented: boolean;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL;


function degToRad(deg: number) {
  return (deg * Math.PI) / 180;
}

function wrap360(deg: number) {
  let v = deg;
  while (v < 0) v += 360;
  while (v >= 360) v -= 360;
  return v;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function resolveDate(date: "03-21" | "06-21" | "12-21") {
  if (date === "03-21") return { month: 3, day: 21 };
  if (date === "12-21") return { month: 12, day: 21 };
  return { month: 6, day: 21 };
}

function dayOfYear(month: number, day: number) {
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return monthDays.slice(0, month - 1).reduce((a, b) => a + b, 0) + day;
}

function solarPositionSimple(
  latitudeDeg: number,
  month: number,
  day: number,
  hourLocal: number
) {
  const n = dayOfYear(month, day);
  const lat = degToRad(latitudeDeg);

  const gamma = (2 * Math.PI / 365) * (n - 1 + (hourLocal - 12) / 24);

  const decl =
    0.006918 -
    0.399912 * Math.cos(gamma) +
    0.070257 * Math.sin(gamma) -
    0.006758 * Math.cos(2 * gamma) +
    0.000907 * Math.sin(2 * gamma) -
    0.002697 * Math.cos(3 * gamma) +
    0.00148 * Math.sin(3 * gamma);

  const hra = degToRad(15 * (hourLocal - 12));

  const sinAlt =
    Math.sin(lat) * Math.sin(decl) +
    Math.cos(lat) * Math.cos(decl) * Math.cos(hra);

  const altitude = Math.asin(Math.max(-1, Math.min(1, sinAlt)));

  const cosAz =
    (Math.sin(decl) - Math.sin(altitude) * Math.sin(lat)) /
    (Math.cos(altitude) * Math.cos(lat) + 1e-9);

  let azimuth = (Math.acos(Math.max(-1, Math.min(1, cosAz))) * 180) / Math.PI;
  if (hourLocal > 12) azimuth = 360 - azimuth;

  return {
    altitudeDeg: (altitude * 180) / Math.PI,
    azimuthDeg: azimuth,
  };
}

function orientationToAzimuth(
  orientation: "North" | "South" | "East" | "West"
) {
  if (orientation === "North") return 0;
  if (orientation === "East") return 90;
  if (orientation === "South") return 180;
  return 270;
}

function sunVectorWorld(altitudeDeg: number, azimuthDeg: number) {
  const sunAlt = degToRad(altitudeDeg);
  const sunAz = degToRad(azimuthDeg);

  return {
    x: Math.sin(sunAz) * Math.cos(sunAlt),
    y: Math.cos(sunAz) * Math.cos(sunAlt),
    z: Math.sin(sunAlt),
  };
}

function getFacadeBasis(facadeAzimuth: number) {
  const fa = degToRad(facadeAzimuth);

  const outward = { x: Math.sin(fa), y: Math.cos(fa) };
  const inward = { x: -outward.x, y: -outward.y };
  const right = { x: Math.cos(fa), y: -Math.sin(fa) };

  return { outward, inward, right };
}

function cross(o: Pt2, a: Pt2, b: Pt2) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHull(points: Pt2[]) {
  if (points.length <= 3) return points;

  const pts = [...points].sort((a, b) =>
    a.x === b.x ? a.y - b.y : a.x - b.x
  );

  const lower: Pt2[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Pt2[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function polygonArea(points: Pt2[]) {
  if (points.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    area += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(area) * 0.5;
}

function computeEffectiveMaxPenetration(
  floorHits: Pt2[],
  roomWidth: number,
  roomDepth: number,
  windowWidth: number,
  cellSize = 0.25
) {
  if (floorHits.length === 0) return 0;

  const rows = new Map<number, Set<number>>();

  for (const p of floorHits) {
    const ix = Math.floor(p.x / cellSize);
    const iy = Math.floor(p.y / cellSize);

    if (!rows.has(iy)) rows.set(iy, new Set<number>());
    rows.get(iy)!.add(ix);
  }

  const minRequiredWidth = Math.max(cellSize, windowWidth * 0.08);

  let maxPenetration = 0;

  for (const [iy, cols] of rows.entries()) {
    const litWidth = cols.size * cellSize;

    if (litWidth + 1e-9 >= minRequiredWidth) {
      maxPenetration = Math.max(maxPenetration, (iy + 1) * cellSize);
    }
  }

  return maxPenetration;
}

function computeEffectivePatchMetrics(
  floorHits: Pt2[],
  roomWidth: number,
  roomDepth: number,
  windowWidth: number,
  cellSize = 0.25
) {
  if (floorHits.length === 0) {
    return { area: 0, maxPenetration: 0 };
  }

  const occupied = new Set<string>();

  for (const p of floorHits) {
    const ix = Math.max(0, Math.min(Math.floor(p.x / cellSize), Math.floor(roomWidth / cellSize)));
    const iy = Math.max(0, Math.min(Math.floor(p.y / cellSize), Math.floor(roomDepth / cellSize)));
    occupied.add(`${ix},${iy}`);
  }

  const rows = new Map<number, number[]>();

  for (const key of occupied) {
    const [ixStr, iyStr] = key.split(",");
    const ix = Number(ixStr);
    const iy = Number(iyStr);

    if (!rows.has(iy)) rows.set(iy, []);
    rows.get(iy)!.push(ix);
  }

  const minRequiredWidth = Math.max(cellSize * 2, windowWidth * 0.12);
  const minRequiredCells = Math.max(2, Math.ceil(minRequiredWidth / cellSize));

  let maxPenetration = 0;

  for (const [iy, colsRaw] of rows.entries()) {
    const cols = [...new Set(colsRaw)].sort((a, b) => a - b);

    let bestRun = 1;
    let run = 1;

    for (let i = 1; i < cols.length; i++) {
      if (cols[i] === cols[i - 1] + 1) {
        run += 1;
      } else {
        bestRun = Math.max(bestRun, run);
        run = 1;
      }
    }

    bestRun = Math.max(bestRun, run);

    if (bestRun >= minRequiredCells) {
      maxPenetration = Math.max(maxPenetration, (iy + 1) * cellSize);
    }
  }

  return {
    area: occupied.size * cellSize * cellSize,
    maxPenetration,
  };
}

function pointInRect(
  px: number,
  py: number,
  rx0: number,
  ry0: number,
  rx1: number,
  ry1: number
) {
  return px >= rx0 && px <= rx1 && py >= ry0 && py <= ry1;
}

function rayHitsHorizontalShade(
  origin: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
  windowLeft: number,
  windowRight: number,
  sillHeight: number,
  windowHeight: number,
  horizontalCount: number,
  horizontalDepth: number,
  horizontalSpacing: number,
  shadingThickness: number
) {
  if (horizontalCount <= 0 || horizontalDepth <= 0 || shadingThickness <= 0) {
    return false;
  }

  for (let i = 0; i < horizontalCount; i++) {
    const zTop = sillHeight + windowHeight - i * horizontalSpacing;
    const zBottom = zTop - shadingThickness;
    if (zBottom < sillHeight) continue;

    const hit = rayIntersectsAABB(
      origin,
      dir,
      { x: windowLeft, y: -horizontalDepth, z: zBottom },
      { x: windowRight, y: 0, z: zTop }
    );

    if (hit) return true;
  }

  return false;
}

function rayIntersectsAABB(
  origin: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
  boxMin: { x: number; y: number; z: number },
  boxMax: { x: number; y: number; z: number }
) {
  const invX = Math.abs(dir.x) > 1e-9 ? 1 / dir.x : Number.POSITIVE_INFINITY;
  const invY = Math.abs(dir.y) > 1e-9 ? 1 / dir.y : Number.POSITIVE_INFINITY;
  const invZ = Math.abs(dir.z) > 1e-9 ? 1 / dir.z : Number.POSITIVE_INFINITY;

  let tmin = -Infinity;
  let tmax = Infinity;

  const tx1 = (boxMin.x - origin.x) * invX;
  const tx2 = (boxMax.x - origin.x) * invX;
  tmin = Math.max(tmin, Math.min(tx1, tx2));
  tmax = Math.min(tmax, Math.max(tx1, tx2));

  const ty1 = (boxMin.y - origin.y) * invY;
  const ty2 = (boxMax.y - origin.y) * invY;
  tmin = Math.max(tmin, Math.min(ty1, ty2));
  tmax = Math.min(tmax, Math.max(ty1, ty2));

  const tz1 = (boxMin.z - origin.z) * invZ;
  const tz2 = (boxMax.z - origin.z) * invZ;
  tmin = Math.max(tmin, Math.min(tz1, tz2));
  tmax = Math.min(tmax, Math.max(tz1, tz2));

  return tmax >= Math.max(tmin, 1e-6);
}

function rayHitsVerticalShade(
  origin: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
  windowLeft: number,
  windowRight: number,
  sillHeight: number,
  windowHeight: number,
  verticalCount: number,
  verticalDepth: number,
  verticalSpacing: number,
  shadingThickness: number
) {
  if (verticalCount <= 0 || verticalDepth <= 0 || shadingThickness <= 0) {
    return false;
  }

  for (let i = 0; i < verticalCount; i++) {
    const x0 = windowLeft + i * verticalSpacing;
    if (x0 >= windowRight) continue;

    const x1 = Math.min(x0 + shadingThickness, windowRight);
    if (x1 <= x0) continue;

    const hit = rayIntersectsAABB(
      origin,
      dir,
      { x: x0, y: -verticalDepth, z: sillHeight },
      { x: x1, y: 0, z: sillHeight + windowHeight }
    );

    if (hit) return true;
  }

  return false;
}

function NumberInput({
  label,
  value,
  setValue,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  setValue: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="mt-3 block">
      <div className="mb-1 text-sm text-slate-600">{label}</div>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => setValue(parseFloat(e.target.value || "0"))}
        step={step}
        min={min}
        max={max}
        className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 outline-none ring-0 focus:border-slate-500"
      />
    </label>
  );
}

function MetricCard({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="text-sm text-slate-500">{title}</div>
      <div className="mt-2 text-xl font-semibold text-slate-900">{value}</div>
    </div>
  );
}

function ScenarioLineChart({
  scenarios,
  metric,
}: {
  scenarios: SavedScenario[];
  metric: "sunlit_area_sqft" | "coverage_ratio" | "max_penetration_ft";
}) {
  const w = 760;
  const h = 340;
  const padL = 64;
  const padR = 24;
  const padT = 28;
  const padB = 88;

  const allHours = Array.from(
    new Set(scenarios.flatMap((s) => s.result.times.map((t) => t.hour)))
  ).sort((a, b) => a - b);

  if (allHours.length === 0) return null;

  const series = scenarios.map((scenario) => {
    const values = allHours.map((hour) => {
      const found = scenario.result.times.find((t) => t.hour === hour);
      if (!found) return 0;

      if (metric === "coverage_ratio") return found.coverage_ratio * 100;
      return found[metric];
    });

    return {
      id: scenario.id,
      name: scenario.name,
      values,
    };
  });

  const allValues = series.flatMap((s) => s.values);
  const maxVal = Math.max(...allValues, 0.01);
  const minVal = 0;

  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const xAt = (hour: number) => {
    if (allHours.length === 1) return padL + plotW / 2;
    const idx = allHours.indexOf(hour);
    return padL + (idx / (allHours.length - 1)) * plotW;
  };

  const yAt = (value: number) => {
    const t = (value - minVal) / (maxVal - minVal || 1);
    return padT + plotH - t * plotH;
  };

  const colors = [
    "#8FB3E8",
    "#A8A29E",
    "#D8A7B1",
    "#A7CDBD",
    "#C7B6E5",
    "#E7BE8A",
  ];

  const yTicks = 5;
  const tickValues = Array.from(
    { length: yTicks + 1 },
    (_, i) => (maxVal / yTicks) * i
  );

  const yAxisLabel =
    metric === "sunlit_area_sqft"
      ? "Glare Area (ft²)"
      : metric === "coverage_ratio"
      ? "Coverage (%)"
      : "Max Penetration (ft)";

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
      <rect x="0" y="0" width={w} height={h} fill="white" />

      {tickValues.map((v, i) => {
        const y = yAt(v);
        return (
          <g key={i}>
            <line
              x1={padL}
              y1={y}
              x2={w - padR}
              y2={y}
              stroke="#e2e8f0"
              strokeWidth="1"
            />
            <text
              x={padL - 10}
              y={y + 4}
              fontSize="11"
              textAnchor="end"
              fill="#64748b"
            >
              {metric === "coverage_ratio" ? `${v.toFixed(1)}%` : v.toFixed(1)}
            </text>
          </g>
        );
      })}

      <line
        x1={padL}
        y1={padT + plotH}
        x2={w - padR}
        y2={padT + plotH}
        stroke="#94a3b8"
        strokeWidth="1.2"
      />
      <line
        x1={padL}
        y1={padT}
        x2={padL}
        y2={padT + plotH}
        stroke="#94a3b8"
        strokeWidth="1.2"
      />

      {allHours.map((hour) => {
        const x = xAt(hour);
        return (
          <g key={hour}>
            <line
              x1={x}
              y1={padT + plotH}
              x2={x}
              y2={padT + plotH + 5}
              stroke="#94a3b8"
              strokeWidth="1"
            />
            <text
              x={x}
              y={padT + plotH + 22}
              fontSize="11"
              textAnchor="middle"
              fill="#64748b"
            >
              {`${hour}:00`}
            </text>
          </g>
        );
      })}

      {series.map((s, idx) => {
        const color = colors[idx % colors.length];
        const points = allHours
          .map((hour, i) => `${xAt(hour)},${yAt(s.values[i])}`)
          .join(" ");

        return (
          <g key={s.id}>
            <polyline
              points={points}
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {allHours.map((hour, i) => (
              <circle
                key={`${s.id}-${hour}`}
                cx={xAt(hour)}
                cy={yAt(s.values[i])}
                r="3.5"
                fill={color}
                stroke="white"
                strokeWidth="1.5"
              />
            ))}
          </g>
        );
      })}

      {/* Y axis label */}
      <text
        x="18"
        y={padT + plotH / 2}
        transform={`rotate(-90 18 ${padT + plotH / 2})`}
        fontSize="12"
        textAnchor="middle"
        fill="#475569"
      >
        {yAxisLabel}
      </text>

      {/* X axis label */}
      <text
        x={padL + plotW / 2}
        y={h - 42}
        fontSize="12"
        textAnchor="middle"
        fill="#475569"
      >
        Time
      </text>

      {/* legend */}
      <g transform={`translate(${padL}, ${h - 16})`}>
        {series.map((s, idx) => {
          const color = colors[idx % colors.length];
          return (
            <g key={s.id} transform={`translate(${idx * 150}, 0)`}>
              <line x1="0" y1="-8" x2="18" y2="-8" stroke={color} strokeWidth="3" />
              <circle cx="9" cy="-8" r="3.5" fill={color} stroke="white" strokeWidth="1.2" />
              <text x="26" y="-4" fontSize="11" fill="#334155">
                {s.name}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
function OrientationDial({
  value,
  onChange,
}: {
  value: number;
  onChange: (deg: number) => void;
}) {
  const size = 180;
  const r = 68;
  const cx = size / 2;
  const cy = size / 2;
  const pointerLen = 52;

  function normalize360(deg: number) {
    let v = deg % 360;
    if (v < 0) v += 360;
    return v;
  }

  function pointerFromDeg(deg: number) {
    const rad = ((deg - 90) * Math.PI) / 180;
    return {
      x: cx + Math.cos(rad) * pointerLen,
      y: cy + Math.sin(rad) * pointerLen,
    };
  }

  function eventToDeg(
    e: React.MouseEvent<SVGSVGElement> | React.PointerEvent<SVGSVGElement>
  ) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const dx = x - cx;
    const dy = y - cy;

    const rad = Math.atan2(dy, dx);
    const deg = (rad * 180) / Math.PI + 90;
    return normalize360(deg);
  }

  const p = pointerFromDeg(value);

  return (
    <div className="mt-3 rounded-2xl border border-slate-300 bg-white p-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm text-slate-600">Orientation</div>
        <div className="text-sm font-medium text-slate-900">
          {Math.round(value)}°
        </div>
      </div>

      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="mx-auto h-44 w-44 cursor-pointer touch-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          onChange(eventToDeg(e));
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) {
            onChange(eventToDeg(e));
          }
        }}
      >
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="white"
          stroke="black"
          strokeWidth="2"
        />

        <line
          x1={cx}
          y1={cy - r}
          x2={cx}
          y2={cy + r}
          stroke="black"
          strokeWidth="1.5"
        />
        <line
          x1={cx - r}
          y1={cy}
          x2={cx + r}
          y2={cy}
          stroke="black"
          strokeWidth="1.5"
        />

        <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="12">
          N
        </text>
        <text x={cx + r + 10} y={cy + 4} textAnchor="middle" fontSize="12">
          E
        </text>
        <text x={cx} y={cy + r + 16} textAnchor="middle" fontSize="12">
          S
        </text>
        <text x={cx - r - 10} y={cy + 4} textAnchor="middle" fontSize="12">
          W
        </text>

        <line
          x1={cx}
          y1={cy}
          x2={p.x}
          y2={p.y}
          stroke="black"
          strokeWidth="7"
          strokeLinecap="round"
        />

        <circle cx={cx} cy={cy} r="4.5" fill="black" />
      </svg>

      <div className="mt-3 flex items-center gap-2">
        <input
          type="range"
          min="0"
          max="359"
          step="1"
          value={Math.round(value)}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full"
        />
        <input
          type="number"
          min="0"
          max="359"
          step="1"
          value={Math.round(value)}
          onChange={(e) => onChange(normalize360(Number(e.target.value) || 0))}
          className="w-20 rounded-lg border border-slate-300 px-2 py-1 text-sm"
        />
      </div>
    </div>
  );
}

export default function Page() {
  const [roomWidth, setRoomWidth] = useState(20);
  const [roomDepth, setRoomDepth] = useState(20);
  const [roomHeight, setRoomHeight] = useState(10);

  const [windowWidth, setWindowWidth] = useState(8);
  const [windowHeight, setWindowHeight] = useState(6);
  const [sillHeight, setSillHeight] = useState(3);
  const [windowOffset, setWindowOffset] = useState(6);

  const [latitude, setLatitude] = useState(47.6);

  const [analysisDate, setAnalysisDate] = useState<"03-21" | "06-21" | "12-21">(
    "06-21"
  );
  const [timeMode, setTimeMode] = useState<"full_day" | "9" | "12" | "15">(
    "full_day"
  );

  const [hasShading, setHasShading] = useState(true);
  const [shadingType, setShadingType] = useState<
    "horizontal" | "vertical" | "eggcrate"
  >("horizontal");

  const [horizontalCount, setHorizontalCount] = useState(3);
  const [verticalCount, setVerticalCount] = useState(3);

  const [horizontalDepth, setHorizontalDepth] = useState(2);
  const [verticalDepth, setVerticalDepth] = useState(2);

  const [shadingThickness, setShadingThickness] = useState(0.25);
  const [horizontalSpacing, setHorizontalSpacing] = useState(1.5);
  const [verticalSpacing, setVerticalSpacing] = useState(1.5);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [previewRotate, setPreviewRotate] = useState(35);
  const [previewZoom, setPreviewZoom] = useState(1);

  const [mounted, setMounted] = useState(false);

  const [orientationDeg, setOrientationDeg] = useState(180);

  useEffect(() => {
  setMounted(true);

  if (typeof window === "undefined") return;

  const raw = localStorage.getItem("savedScenarios");
  if (!raw) return;

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      setSavedScenarios(parsed);
    }
  } catch (e) {
    console.error("Failed to load saved scenarios from localStorage:", e);
  }
}, []);

  const [scenarioName, setScenarioName] = useState("");
  const [savedScenarios, setSavedScenarios] = useState<SavedScenario[]>([]);
  const [selectedScenarioIds, setSelectedScenarioIds] = useState<string[]>([]);
  const [comparisonMetric, setComparisonMetric] = useState<
    "sunlit_area_sqft" | "coverage_ratio" | "max_penetration_ft"
  >("sunlit_area_sqft");
  function toggleScenarioSelection(id: string) {
    setSelectedScenarioIds((prev) =>
      prev.includes(id)
        ? prev.filter((item) => item !== id) // 取消选中
        : [...prev, id] // 选中
    );
  }

  const dragRef = useRef<{ dragging: boolean; lastX: number }>({
    dragging: false,
    lastX: 0,
  });

  const frontPreviewRef = useRef<HTMLDivElement | null>(null);
  const topPreviewRef = useRef<HTMLDivElement | null>(null);
  const box3DPreviewRef = useRef<HTMLDivElement | null>(null);
  const comparisonRef = useRef<HTMLDivElement | null>(null);

  const SHADOW_FILL = "#d1d5db";
  const SHADOW_OPACITY = 0.9;

  const displayHour =
    timeMode === "9" ? 9 : timeMode === "12" ? 12 : timeMode === "15" ? 15 : 12;

  const { month, day } = resolveDate(analysisDate);
  const { altitudeDeg, azimuthDeg } = solarPositionSimple(
    latitude,
    month,
    day,
    displayHour
  );
  const facadeAzimuth = orientationDeg;

  async function runAnalysis() {
    setLoading(true);
    setError(null);

    try {
      const payload = {
        roomWidth,
        roomDepth,
        roomHeight,
        windowWidth,
        windowHeight,
        sillHeight,
        windowOffset,
        latitude,
        orientationDeg,
        analysisDate,
        timeMode,
        hasShading,
        shadingType,
        shadingThickness,
        horizontalCount,
        verticalCount,
        horizontalDepth,
        verticalDepth,
        horizontalSpacing,
        verticalSpacing,
      };

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || "Analysis failed.");
      }

      setResult(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error.";
      setError(msg);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function orientPlanPoint(
  x: number,
  y: number,
  roomWidth: number,
  roomDepth: number,
  orientation: "North" | "South" | "East" | "West"
) {
  if (orientation === "South") {
    return { x, y };
  }

  if (orientation === "North") {
    return { x: roomWidth - x, y: roomDepth - y };
  }

  if (orientation === "East") {
    return { x: roomWidth - y, y: x };
  }

  // West
  return { x: y, y: roomDepth - x };
}

  function buildScenarioName() {
    if (scenarioName.trim()) return scenarioName.trim();
    if (!hasShading) return "No Shading";
    if (shadingType === "horizontal") return `Horizontal ${horizontalDepth}ft`;
    if (shadingType === "vertical") return `Vertical ${verticalDepth}ft`;
    return `Eggcrate ${horizontalDepth}ft / ${verticalDepth}ft`;
  }

  function saveCurrentScenario() {
    if (!result) return;

    const newScenario: SavedScenario = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      name: buildScenarioName(),
      inputs: {
        roomWidth,
        roomDepth,
        roomHeight,
        windowWidth,
        windowHeight,
        sillHeight,
        windowOffset,
        latitude,
        orientation,
        analysisDate,
        timeMode,
        hasShading,
        shadingType,
        horizontalCount,
        verticalCount,
        horizontalDepth,
        verticalDepth,
        shadingThickness,
        horizontalSpacing,
        verticalSpacing,
      },
      result,
    };

    setSavedScenarios((prev) => {
      const updated = [newScenario, ...prev];
      if (typeof window !== "undefined") {
        localStorage.setItem("savedScenarios", JSON.stringify(updated));
      }
      return updated;
    });

    setSelectedScenarioIds((prev) =>
      prev.includes(newScenario.id) ? prev : [newScenario.id, ...prev]
    );

    setScenarioName("");

    if (typeof window !== "undefined") {
      setTimeout(() => {
        const el = document.getElementById("saved-scenarios");
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    }
  }

  function deleteScenario(id: string) {
    setSavedScenarios((prev) => {
      const updated = prev.filter((s) => s.id !== id);
      if (typeof window !== "undefined") {
        localStorage.setItem("savedScenarios", JSON.stringify(updated));
      }
      return updated;
    });

    setSelectedScenarioIds((prev) => prev.filter((item) => item !== id));
  }

  async function exportElementAsPng(
    el: HTMLDivElement | null,
    filename: string
  ) {
    if (!el) return;

    const dataUrl = await toPng(el, {
      cacheBust: true,
      pixelRatio: 2,
      backgroundColor: "#ffffff",
    });

    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    a.click();
  }


  async function exportSvgContainerAsPng(
    container: HTMLDivElement | null,
    filename: string,
    scale = 2
  ) {
    if (!container) return;

    const svg = container.querySelector("svg");
    if (!svg) return;

    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svg);

    const viewBox = svg.getAttribute("viewBox");
    if (!viewBox) return;

    const [, , vbWidth, vbHeight] = viewBox.split(" ").map(Number);
    const width = vbWidth * scale;
    const height = vbHeight * scale;

    const blob = new Blob([svgString], {
      type: "image/svg+xml;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        return;
      }

      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);

      const pngUrl = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = filename;
      a.click();

      URL.revokeObjectURL(url);
    };

    img.src = url;
  }

  async function exportAllPreviews() {
    await exportSvgContainerAsPng(frontPreviewRef.current, "front-preview.png");
    await exportSvgContainerAsPng(topPreviewRef.current, "top-preview.png");
    await exportSvgContainerAsPng(box3DPreviewRef.current, "3d-preview.png");

    if (comparisonRef.current) {
      await exportElementAsPng(comparisonRef.current, "scenario-comparison.png");
    }
  }
  function computeSunPatchAtHour(hour: number, useShading: boolean): SunPatchResult | null {
  const { month, day } = resolveDate(analysisDate);
  const { altitudeDeg: alt, azimuthDeg: az } = solarPositionSimple(
    latitude,
    month,
    day,
    hour
  );

  if (alt <= 0) return null;

  const sun = sunVectorWorld(alt, az);
  const { inward, right } = getFacadeBasis(facadeAzimuth);

  const sx = sun.x * right.x + sun.y * right.y;
  const sy = sun.x * inward.x + sun.y * inward.y;
  const sz = sun.z;

  const rayDirIntoRoom = {
  x: -sx,
  y: -sy,
  z: -sz,
  };

  const rayToSun = {
    x: sx,
    y: sy,
    z: sz,
  };

  if (rayDirIntoRoom.y <= 0 || rayDirIntoRoom.z >= 0) return null;

  const windowLeft = clamp(windowOffset, 0, Math.max(0, roomWidth - windowWidth));
  const windowRight = windowLeft + windowWidth;

  const xSamples = 60;
  const zSamples = 40;
  const totalSamples = xSamples * zSamples;

  const floorHits: Pt2[] = [];
  let visibleCount = 0;

  for (let ix = 0; ix < xSamples; ix++) {
    for (let iz = 0; iz < zSamples; iz++) {
      const px = windowLeft + ((ix + 0.5) / xSamples) * windowWidth;
      const pz = sillHeight + ((iz + 0.5) / zSamples) * windowHeight;

      const origin = { x: px, y: 0, z: pz };
      let blocked = false;

      if (useShading && hasShading) {
        const originOutside = {
          x: origin.x,
          y: -1e-4,
          z: origin.z,
        };

        if (shadingType === "horizontal" || shadingType === "eggcrate") {
          blocked = rayHitsHorizontalShade(
            originOutside,
            rayToSun,
            windowLeft,
            windowRight,
            sillHeight,
            windowHeight,
            horizontalCount,
            horizontalDepth,
            horizontalSpacing,
            shadingThickness
          );
        }

        if (!blocked && (shadingType === "vertical" || shadingType === "eggcrate")) {
          blocked = rayHitsVerticalShade(
            originOutside,
            rayToSun,
            windowLeft,
            windowRight,
            sillHeight,
            windowHeight,
            verticalCount,
            verticalDepth,
            verticalSpacing,
            shadingThickness
          );
        }
      }

      if (blocked) continue;

      const t = (0 - origin.z) / rayDirIntoRoom.z;
      if (t <= 0) continue;

      const hitX = origin.x + rayDirIntoRoom.x * t;
      const hitY = origin.y + rayDirIntoRoom.y * t;

      if (hitY < 0 || hitY > roomDepth) continue;
      if (hitX < 0 || hitX > roomWidth) continue;

      visibleCount += 1;
      floorHits.push({
        x: clamp(hitX, 0, roomWidth),
        y: clamp(hitY, 0, roomDepth),
      });
    }
  }

  if (floorHits.length < 3) return null;

  const hull = convexHull(floorHits);
  const xs = floorHits.map((p) => p.x);
  const ys = floorHits.map((p) => p.y);

  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);

  const metrics = computeEffectivePatchMetrics(
    floorHits,
    roomWidth,
    roomDepth,
    windowWidth,
    0.25
  );

  const maxPenetration = metrics.maxPenetration;
  const area = metrics.area;

  return {
    x0,
    x1,
    y0,
    y1,
    area,
    maxPenetration,
    floorPoints: floorHits,
    hullPoints: hull,
    altitudeDeg: alt,
    azimuthDeg: az,
    segmented: shadingType === "vertical" || shadingType === "eggcrate",
  };
}

  

  const maxPenetrationInfo = useMemo(() => {
    if (!result || !result.times || result.times.length === 0) return null;
    const best = result.times.reduce((max, item) =>
      item.max_penetration_ft > max.max_penetration_ft ? item : max
    );
    return {
      value: best.max_penetration_ft,
      label: best.label,
    };
  }, [result]);

  const daylightHours = useMemo(() => {
  if (!result?.times?.length) return [];

  const data = result.times
    .filter((t) => t.sun_visible)
    .map((t) => {
      const patch = computeSunPatchAtHour(t.hour, hasShading);
      return {
        hour: t.hour,
        label: t.label,
        patch,
        penetration: patch?.maxPenetration ?? 0,
        area: patch?.area ?? 0,
      };
    })
    .filter((item) => item.patch !== null);

  console.log("result.times", result?.times);
  console.log("daylightHours hours =", data.map((d) => d.hour));

  return data;
}, [
    result,
    hasShading,
    shadingType,
    horizontalDepth,
    verticalDepth,
    horizontalCount,
    verticalCount,
    horizontalSpacing,
    verticalSpacing,
    shadingThickness,
    roomWidth,
    roomDepth,
    roomHeight,
    windowWidth,
    windowHeight,
    sillHeight,
    latitude,
    analysisDate,
    facadeAzimuth,
  ]);

  const selectedScenarios = useMemo(() => {
    return savedScenarios.filter((s) => selectedScenarioIds.includes(s.id));
  }, [savedScenarios, selectedScenarioIds]);

  const geometricAverageCoverage = useMemo(() => {
    if (!daylightHours.length) return null;
    const avgArea =
      daylightHours.reduce((sum, item) => sum + (item.area ?? 0), 0) /
      daylightHours.length;
    const roomArea = roomWidth * roomDepth;
    return roomArea > 0 ? (avgArea / roomArea) * 100 : null;
  }, [daylightHours, roomWidth, roomDepth]);

  const frontPreview = useMemo(() => {
    const w = 320;
    const h = 240;
    const pad = 24;

    const sx = (value: number) => pad + (value / roomWidth) * (w - pad * 2);
    const sy = (value: number) => h - pad - (value / roomHeight) * (h - pad * 2);

    const wallLeft = sx(0);
    const wallRight = sx(roomWidth);
    const wallTop = sy(roomHeight);
    const wallBottom = sy(0);

    const windowLeftVal = clamp(windowOffset, 0, Math.max(0, roomWidth - windowWidth));
    const windowRightVal = windowLeftVal + windowWidth;

    const winLeft = sx(windowLeftVal);
    const winRight = sx(windowRightVal);
    const winTop = sy(sillHeight + windowHeight);
    const winBottom = sy(sillHeight);

    const lines: React.ReactNode[] = [];

    if (hasShading && (shadingType === "horizontal" || shadingType === "eggcrate")) {
  for (let i = 0; i < horizontalCount; i++) {
    const z = sillHeight + windowHeight - i * horizontalSpacing;
    if (z < sillHeight) break;

    const thickPx = Math.max(
      2,
      (shadingThickness / roomHeight) * (h - pad * 2)
    );

    lines.push(
      <rect
        key={`h-${i}`}
        x={winLeft}
        y={sy(z)}
        width={winRight - winLeft}
        height={thickPx}
        fill="#64748b"
        opacity={0.9}
      />
    );
  }
}

  if (hasShading && (shadingType === "vertical" || shadingType === "eggcrate")) {
  for (let i = 0; i < verticalCount; i++) {
    const x = windowLeftVal + i * verticalSpacing;
    if (x > windowRightVal) break;

    const thickPxX = Math.max(
      2,
      (shadingThickness / roomWidth) * (w - pad * 2)
    );

    lines.push(
      <rect
        key={`v-${i}`}
        x={sx(x)}
        y={winTop}
        width={thickPxX}
        height={winBottom - winTop}
        fill="#64748b"
        opacity={0.9}
      />
    );
  }
}

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
        <rect
          x={wallLeft}
          y={wallTop}
          width={wallRight - wallLeft}
          height={wallBottom - wallTop}
          fill="none"
          stroke="black"
          strokeWidth="1.5"
        />

        <rect
          x={winLeft}
          y={winTop}
          width={winRight - winLeft}
          height={winBottom - winTop}
          fill="none"
          stroke="black"
          strokeWidth="1.5"
        />

        {lines}

        <text x="12" y="18" fontSize="12" fill="black">
          Elevation
        </text>
      </svg>
    );
    }, [
    roomWidth,
    roomHeight,
    windowWidth,
    windowHeight,
    sillHeight,
    windowOffset,
    hasShading,
    shadingType,
    horizontalCount,
    verticalCount,
    horizontalSpacing,
    verticalSpacing,
  ]);
  const topPreview = useMemo(() => {
    const w = 320;
    const h = 240;
    const pad = 40;

    const innerW = w - pad * 2;
    const innerH = h - pad * 2;
    const scale = Math.min(innerW / roomWidth, innerH / roomDepth);

    const offsetX = (w - roomWidth * scale) / 2;
    const offsetY = (h - roomDepth * scale) / 2;

    const sx = (value: number) => offsetX + value * scale;
    const sy = (value: number) => offsetY + value * scale;

    const roomLeft = sx(0);
    const roomRight = sx(roomWidth);
    const roomTop = sy(0);
    const roomBottom = sy(roomDepth);

    const windowLeftVal = clamp(windowOffset, 0, Math.max(0, roomWidth - windowWidth));
    const windowRightVal = windowLeftVal + windowWidth;
    const windowCenter = windowLeftVal + windowWidth / 2;

    const winLeft = sx(windowLeftVal);
    const winRight = sx(windowRightVal);

    const shapes: React.ReactNode[] = [];

    function topFacadeLine(u0: number, u1: number) {
      if (orientation === "South") {
        return {
          x1: sx(u0),
          y1: roomTop,
          x2: sx(u1),
          y2: roomTop,
        };
      }
      if (orientation === "North") {
        return {
          x1: sx(roomWidth - u0),
          y1: roomBottom,
          x2: sx(roomWidth - u1),
          y2: roomBottom,
        };
      }
      if (orientation === "East") {
        return {
          x1: roomRight,
          y1: sy(u0),
          x2: roomRight,
          y2: sy(u1),
        };
      }
      return {
        x1: roomLeft,
        y1: sy(roomDepth - u0),
        x2: roomLeft,
        y2: sy(roomDepth - u1),
      };
    }

    if (hasShading && (shadingType === "horizontal" || shadingType === "eggcrate")) {
      const d = Math.max(8, horizontalDepth * scale);

      if (orientation === "South") {
        shapes.push(
          <rect
            key="h-top"
            x={winLeft}
            y={roomTop - d}
            width={winRight - winLeft}
            height={d}
            fill="none"
            stroke="black"
            strokeWidth="1.2"
          />
        );
      } else if (orientation === "North") {
        shapes.push(
          <rect
            key="h-top"
            x={sx(roomWidth - windowRightVal)}
            y={roomBottom}
            width={winRight - winLeft}
            height={d}
            fill="none"
            stroke="black"
            strokeWidth="1.2"
          />
        );
      } else if (orientation === "East") {
        shapes.push(
          <rect
            key="h-top"
            x={roomRight}
            y={sy(windowLeftVal)}
            width={d}
            height={sy(windowRightVal) - sy(windowLeftVal)}
            fill="none"
            stroke="black"
            strokeWidth="1.2"
          />
        );
      } else {
        shapes.push(
          <rect
            key="h-top"
            x={roomLeft - d}
            y={sy(roomDepth - windowRightVal)}
            width={d}
            height={sy(roomDepth - windowLeftVal) - sy(roomDepth - windowRightVal)}
            fill="none"
            stroke="black"
            strokeWidth="1.2"
          />
        );
      }
    }

    if (hasShading && (shadingType === "vertical" || shadingType === "eggcrate")) {
      const d = Math.max(8, verticalDepth * scale);
      const thick = Math.max(2, shadingThickness * scale);

      for (let i = 0; i < verticalCount; i++) {
        const x0 = windowLeftVal + i * verticalSpacing;
        if (x0 > windowRightVal) break;

        if (orientation === "South") {
          shapes.push(
            <rect
              key={`vt-${i}`}
              x={sx(x0)}
              y={roomTop - d}
              width={thick}
              height={d}
              fill="none"
              stroke="black"
              strokeWidth="1.2"
            />
          );
        } else if (orientation === "North") {
          shapes.push(
            <rect
              key={`vt-${i}`}
              x={sx(roomWidth - x0) - thick}
              y={roomBottom}
              width={thick}
              height={d}
              fill="none"
              stroke="black"
              strokeWidth="1.2"
            />
          );
        } else if (orientation === "East") {
          shapes.push(
            <rect
              key={`vt-${i}`}
              x={roomRight}
              y={sy(x0)}
              width={d}
              height={thick}
              fill="none"
              stroke="black"
              strokeWidth="1.2"
            />
          );
        } else {
          shapes.push(
            <rect
              key={`vt-${i}`}
              x={roomLeft - d}
              y={sy(roomDepth - x0) - thick}
              width={d}
              height={thick}
              fill="none"
              stroke="black"
              strokeWidth="1.2"
            />
          );
        }
      }
    }

    const activeHour = displayHour;
    const singlePatch = computeSunPatchAtHour(activeHour, hasShading);

    const patchShapes: React.ReactNode[] = [];
    const penetrationGraphics: React.ReactNode[] = [];

    if (timeMode === "full_day" && daylightHours.length > 0) {
      const total = daylightHours.length;

      daylightHours.forEach((item, idx) => {
        if (!item.patch) return;
        const p = item.patch;

        if (p.hullPoints.length >= 3) {
          const opacity = 0.18 + (idx / Math.max(1, total - 1)) * 0.42;

          patchShapes.push(
            <polygon
              key={`plan-poly-${idx}`}
              points={p.hullPoints.map((pt) => `${sx(pt.x)},${sy(pt.y)}`).join(" ")}
              fill={SHADOW_FILL}
              opacity={opacity}
            />
          );
        }
      });

      const visible = [...daylightHours].sort((a, b) => a.hour - b.hour);
      const earliest = visible[0];
      const latest = visible[visible.length - 1];
      const noon = visible.find((d) => d.hour === 12) ?? null;
      const maxItem = visible.reduce((max, item) =>
        item.penetration > max.penetration ? item : max
      );

      const selectedMarkers = [earliest, noon, latest, maxItem]
        .filter(Boolean)
        .filter(
          (item, index, arr) =>
            arr.findIndex((d) => d?.hour === item?.hour) === index
        );

      selectedMarkers.forEach((item, idx) => {
        if (!item) return;

        const lineEnd = Math.min(roomDepth, item.penetration);
        const y = sy(lineEnd);

        const markerHalf = item.hour === maxItem.hour ? 18 : 12;
        const x1 = sx(windowCenter) - markerHalf;
        const x2 = sx(windowCenter) + markerHalf;
        const labelOffsetY = [-8, 12, -14, 18][idx] ?? (idx % 2 === 0 ? -8 : 12);
        const isMax = item.hour === maxItem.hour;

        penetrationGraphics.push(
          <g key={`pen-marker-${item.hour}`}>
            <line
              x1={x1}
              y1={y}
              x2={x2}
              y2={y}
              stroke="#6b7280"
              strokeWidth={isMax ? "2.2" : "1.3"}
              strokeDasharray={isMax ? "none" : "4 3"}
              opacity={0.95}
            />
            <text
              x={x2 + 12}
              y={y + labelOffsetY}
              fontSize="9"
              fill="#4b5563"
              opacity={0.98}
            >
              {isMax ? `${item.label} max` : item.label}
            </text>
          </g>
        );
      });
    } else if (singlePatch) {
      if (singlePatch.hullPoints.length >= 3) {
        patchShapes.push(
          <polygon
            key="plan-patch-main"
            points={singlePatch.hullPoints
              .map((pt) => `${sx(pt.x)},${sy(pt.y)}`)
              .join(" ")}
            fill={SHADOW_FILL}
            opacity={SHADOW_OPACITY}
          />
        );
      }

      if (maxPenetrationInfo && maxPenetrationInfo.value > 0) {
        const lineEnd = Math.min(roomDepth, singlePatch.maxPenetration);
        penetrationGraphics.push(
          <g key="single-pen">
            <line
              x1={sx(windowCenter)}
              y1={sy(0)}
              x2={sx(windowCenter)}
              y2={sy(lineEnd)}
              stroke="#6b7280"
              strokeWidth="2"
              strokeDasharray="6 4"
            />
            <text
              x={sx(windowCenter) + 8}
              y={sy(lineEnd) - 6}
              fontSize="10"
              fill="#4b5563"
            >
              {singlePatch.maxPenetration.toFixed(1)} ft @ {displayHour}:00
            </text>
          </g>
        );
      }
    }

   const windowLine = topFacadeLine(windowLeftVal, windowRightVal);

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
        <rect
          x={roomLeft}
          y={roomTop}
          width={roomRight - roomLeft}
          height={roomBottom - roomTop}
          fill="none"
          stroke="black"
          strokeWidth="1.5"
        />

        {patchShapes}
        {penetrationGraphics}

        <line
          x1={windowLine.x1}
          y1={windowLine.y1}
          x2={windowLine.x2}
          y2={windowLine.y2}
          stroke="black"
          strokeWidth="2"
        />

        {shapes}

        <text x="12" y="18" fontSize="12" fill="black">
          Top view
        </text>
        {timeMode === "full_day" && (
          <text x="12" y="30" fontSize="9" fill="#6b7280">
            all day: earliest / noon / latest / max
          </text>
        )}
      </svg>
    );
   }, [
    roomWidth,
    roomDepth,
    windowWidth,
    windowOffset,
    hasShading,
    shadingType,
    horizontalDepth,
    verticalDepth,
    verticalCount,
    verticalSpacing,
    shadingThickness,
    displayHour,
    maxPenetrationInfo,
    daylightHours,
    timeMode,
    orientation,
  ]);

  const box3DPreview = useMemo(() => {
    const w = 920;
    const h = 560;

    const originX = 360;
    const originY = 390;

    const domeRadius = 25.5;
    const domeCenterX = roomWidth / 2;
    const domeCenterY = roomDepth * 0.55;
    const domeCenterZ = 0;

    const zoom = 16 * previewZoom;
    const yaw = degToRad(previewRotate + orientationDeg);

    const roomScaleY = 0.8;
    const roomScaleZ = 0.95;

    const roomX = roomWidth * zoom;
    const roomY = roomDepth * zoom * roomScaleY;
    const roomZ = roomHeight * zoom * roomScaleZ;

    const winW = windowWidth * zoom;
    const winH = windowHeight * zoom * roomScaleZ;
    const sill = sillHeight * zoom * roomScaleZ;
    const winOffsetX =
      clamp(windowOffset, 0, Math.max(0, roomWidth - windowWidth)) * zoom;

    function project(x: number, y: number, z: number) {
      const cosA = Math.cos(yaw);
      const sinA = Math.sin(yaw);

      const rx = x * cosA - y * sinA;
      const ry = x * sinA + y * cosA;

      return {
        x: originX + rx,
        y: originY + ry * 0.4 - z,
        depth: ry,
      };
    }

    function pts(p: { x: number; y: number }[]) {
      return p.map((pt) => `${pt.x},${pt.y}`).join(" ");
    }

    function depth(p: { depth: number }[]) {
      return p.reduce((a, b) => a + b.depth, 0) / p.length;
    }

    // Rotate facade geometry itself to the correct wall
    function facadePoint(u: number, outwardDepth: number, z: number) {
      // outwardDepth: 0 = facade plane, positive = outside the room
      if (orientation === "South") {
        return project(u, 0 - outwardDepth, z);
      }
      if (orientation === "North") {
        return project(roomX - u, roomY + outwardDepth, z);
      }
      if (orientation === "East") {
        return project(roomX + outwardDepth, u, z);
      }
      // West
      return project(0 - outwardDepth, roomY - u, z);
    }

    function worldToPreviewSun(
      altitudeDegLocal: number,
      azimuthDegLocal: number,
      radius: number
    ) {
      const sun = sunVectorWorld(altitudeDegLocal, azimuthDegLocal);
      const { inward, right } = getFacadeBasis(facadeAzimuth);

      const sx = sun.x * right.x + sun.y * right.y;
      const sy = sun.x * inward.x + sun.y * inward.y;
      const sz = sun.z;

      const px = (domeCenterX + sx * radius) * zoom;
      const py = (domeCenterY + sy * radius * roomScaleY) * zoom;
      const pz = (domeCenterZ + sz * radius * roomScaleZ) * zoom;

      return project(px, py, pz);
    }

    function seasonalSunPath(dateKey: "03-21" | "06-21" | "12-21") {
      const { month, day } = resolveDate(dateKey);
      const pts3d: { x: number; y: number }[] = [];

      for (let hour = 5; hour <= 19; hour += 0.25) {
        const sp = solarPositionSimple(latitude, month, day, hour);
        if (sp.altitudeDeg <= 0) continue;

        const p = worldToPreviewSun(sp.altitudeDeg, sp.azimuthDeg, domeRadius);
        pts3d.push({ x: p.x, y: p.y });
      }

      return pts3d.map((p) => `${p.x},${p.y}`).join(" ");
    }

    function activeSunPoint() {
      return worldToPreviewSun(altitudeDeg, azimuthDeg, domeRadius);
    }

    function groundDomeRingPoints(radius: number) {
      const pts3d: { x: number; y: number }[] = [];

      for (let i = 0; i <= 120; i++) {
        const t = (i / 120) * Math.PI * 2;
        const gx = domeCenterX + Math.sin(t) * radius;
        const gy = domeCenterY + Math.cos(t) * radius * 0.72;
        const p = project(gx * zoom, gy * zoom * roomScaleY, 0);
        pts3d.push({
          x: Number(p.x.toFixed(2)),
          y: Number(p.y.toFixed(2)),
        });
      }

      return pts3d.map((p) => `${p.x},${p.y}`).join(" ");
    }

    function pathLabelPoint(
      dateKey: "03-21" | "06-21" | "12-21",
      hour: number,
      radius = domeRadius
    ) {
      const { month, day } = resolveDate(dateKey);
      const sp = solarPositionSimple(latitude, month, day, hour);
      return worldToPreviewSun(sp.altitudeDeg, sp.azimuthDeg, radius);
    }

    function groundCompassPoint(azimuthDegLocal: number, radius: number) {
      const azRad = degToRad(azimuthDegLocal);
      const gx = roomWidth / 2 + Math.sin(azRad) * radius;
      const gy = roomDepth * 0.55 + Math.cos(azRad) * radius * 0.72;
      return project(gx * zoom, gy * zoom * roomScaleY, 0);
    }

    // Room box stays as-is
    const A = project(0, 0, 0);
    const B = project(roomX, 0, 0);
    const C = project(roomX, roomY, 0);
    const D = project(0, roomY, 0);

    const A1 = project(0, 0, roomZ);
    const B1 = project(roomX, 0, roomZ);
    const C1 = project(roomX, roomY, roomZ);
    const D1 = project(0, roomY, roomZ);

    // Window now actually moves to the oriented facade
    const W1 = facadePoint(winOffsetX, 0, sill);
    const W2 = facadePoint(winOffsetX + winW, 0, sill);
    const W3 = facadePoint(winOffsetX + winW, 0, sill + winH);
    const W4 = facadePoint(winOffsetX, 0, sill + winH);

    const faces = [
      { pts: [A, B, C, D], fill: "#f8fafc", d: depth([A, B, C, D]) },
      { pts: [A, D, D1, A1], fill: "#f1f5f9", d: depth([A, D, D1, A1]) },
      { pts: [B, C, C1, B1], fill: "#e2e8f0", d: depth([B, C, C1, B1]) },
      { pts: [A, B, B1, A1], fill: "#ffffff", d: depth([A, B, B1, A1]) },
    ].sort((a, b) => a.d - b.d);

    const patch =
      timeMode === "full_day"
        ? null
        : computeSunPatchAtHour(displayHour, hasShading);

    const patchShapes: React.ReactNode[] = [];
    if (patch && patch.hullPoints.length >= 3) {
      const poly3d = patch.hullPoints.map((pt) =>
        project(pt.x * zoom, pt.y * zoom * roomScaleY, 0)
      );

      patchShapes.push(
        <polygon
          key="patch-3d-main"
          points={pts(poly3d)}
          fill={SHADOW_FILL}
          opacity={SHADOW_OPACITY}
        />
      );
    }

    const allDayFloorPatches3D: React.ReactNode[] = [];
    if (timeMode === "full_day" && daylightHours.length > 0) {
      const total = daylightHours.length;

      daylightHours.forEach((item, idx) => {
        if (!item.patch || item.patch.hullPoints.length < 3) return;

        const opacity = 0.1 + (idx / Math.max(1, total - 1)) * 0.22;

        const points = item.patch.hullPoints
          .map((pt) => {
            const p = project(pt.x * zoom, pt.y * zoom * roomScaleY, 0);
            return `${p.x},${p.y}`;
          })
          .join(" ");

        allDayFloorPatches3D.push(
          <polygon
            key={`box-allday-${item.hour}`}
            points={points}
            fill={SHADOW_FILL}
            opacity={opacity}
          />
        );
      });
    }

    const shadeFaces3D: React.ReactNode[] = [];
    const shadingGroups: React.ReactNode[] = [];

    // Horizontal shading
    if (hasShading && (shadingType === "horizontal" || shadingType === "eggcrate")) {
      const x0 = ((roomWidth - windowWidth) / 2) * zoom;
      const x1 = ((roomWidth + windowWidth) / 2) * zoom;

      for (let i = 0; i < horizontalCount; i++) {
        const zTopFt = sillHeight + windowHeight - i * horizontalSpacing;
        const zBottomFt = zTopFt - shadingThickness;
        if (zBottomFt < sillHeight) continue;

        const zTop = zTopFt * zoom * roomScaleZ;
        const zBottom = zBottomFt * zoom * roomScaleZ;

        const yFront = 0;
        const yBack = horizontalDepth * zoom * roomScaleY;

        const p1 = facadePoint(x0, yFront, zTop);
        const p2 = facadePoint(x1, yFront, zTop);
        const p3 = facadePoint(x1, yBack, zTop);
        const p4 = facadePoint(x0, yBack, zTop);

        const p5 = facadePoint(x0, yFront, zBottom);
        const p6 = facadePoint(x1, yFront, zBottom);
        const p7 = facadePoint(x1, yBack, zBottom);
        const p8 = facadePoint(x0, yBack, zBottom);

        shadeFaces3D.push(
          <g key={`hshade-${i}`}>
            <polygon
              points={pts([p1, p2, p3, p4])}
              fill="#94a3b8"
              opacity={0.9}
              stroke="#475569"
              strokeWidth="0.8"
            />
            <polygon
              points={pts([p4, p3, p7, p8])}
              fill="#64748b"
              opacity={0.92}
              stroke="#475569"
              strokeWidth="0.8"
            />
            <polygon
              points={pts([p2, p3, p7, p6])}
              fill="#7c8ea3"
              opacity={0.9}
              stroke="#475569"
              strokeWidth="0.8"
            />
          </g>
        );

        shadingGroups.push(
          <g key={`hf-${i}`}>
            <polygon points={pts([p1, p2, p6, p5])} fill="#64748b" opacity={0.95} />
            <polygon points={pts([p2, p3, p7, p6])} fill="#475569" opacity={0.95} />
            <polygon points={pts([p5, p6, p7, p8])} fill="#94a3b8" opacity={0.95} />
          </g>
        );
      }
    }

    // Vertical shading
    if (hasShading && (shadingType === "vertical" || shadingType === "eggcrate")) {
      for (let i = 0; i < verticalCount; i++) {
        const x0Ft = (roomWidth - windowWidth) / 2 + i * verticalSpacing;
        const x1Ft = Math.min(x0Ft + shadingThickness, (roomWidth + windowWidth) / 2);
        if (x0Ft >= (roomWidth + windowWidth) / 2) break;

        const x0 = x0Ft * zoom;
        const x1 = x1Ft * zoom;
        const z0 = sillHeight * zoom * roomScaleZ;
        const z1 = (sillHeight + windowHeight) * zoom * roomScaleZ;
        const depthVal = verticalDepth * zoom * roomScaleY;

        const p1 = facadePoint(x0, 0, z0);
        const p2 = facadePoint(x1, 0, z0);
        const p3 = facadePoint(x1, depthVal, z0);
        const p4 = facadePoint(x0, depthVal, z0);

        const p5 = facadePoint(x0, 0, z1);
        const p6 = facadePoint(x1, 0, z1);
        const p7 = facadePoint(x1, depthVal, z1);
        const p8 = facadePoint(x0, depthVal, z1);

        shadeFaces3D.push(
          <g key={`vshade-${i}`}>
            <polygon
              points={pts([p1, p2, p6, p5])}
              fill="#94a3b8"
              opacity={0.9}
              stroke="#475569"
              strokeWidth="0.8"
            />
            <polygon
              points={pts([p2, p3, p7, p6])}
              fill="#64748b"
              opacity={0.92}
              stroke="#475569"
              strokeWidth="0.8"
            />
            <polygon
              points={pts([p4, p3, p7, p8])}
              fill="#7c8ea3"
              opacity={0.9}
              stroke="#475569"
              strokeWidth="0.8"
            />
          </g>
        );

        shadingGroups.push(
          <g key={`vf-${i}`}>
            <polygon points={pts([p1, p2, p6, p5])} fill="#64748b" opacity={0.95} />
            <polygon points={pts([p2, p3, p7, p6])} fill="#475569" opacity={0.95} />
            <polygon points={pts([p5, p6, p7, p8])} fill="#94a3b8" opacity={0.95} />
          </g>
        );
      }
    }

    const sunPt = activeSunPoint();
    const juneLabelPt = pathLabelPoint("06-21", 12);
    const marchLabelPt = pathLabelPoint("03-21", 12);
    const decLabelPt = pathLabelPoint("12-21", 12);

    const northPt = groundCompassPoint(0, 50.5);
    const eastPt = groundCompassPoint(90, 50.5);
    const southPt = groundCompassPoint(180, 50.5);
    const westPt = groundCompassPoint(270, 50.5);

    const domeRingPoints = groundDomeRingPoints(domeRadius);

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
        {faces.map((face, i) => (
          <polygon
            key={i}
            points={pts(face.pts)}
            fill={face.fill}
            stroke="#94a3b8"
            strokeWidth="1.5"
          />
        ))}

        {patchShapes}

        <polygon
          points={pts([W1, W2, W3, W4])}
          fill="#dbeafe"
          stroke="#3b82f6"
          strokeWidth="1.5"
        />

        <polygon
          points={domeRingPoints}
          fill="#e5e7eb"
          opacity={0.28}
          stroke="#d1d5db"
          strokeWidth="1.1"
        />

        {allDayFloorPatches3D}

        <polyline
          points={domeRingPoints}
          fill="none"
          stroke="#ffffff"
          strokeWidth="1.4"
          strokeDasharray="7 7"
          opacity={0.95}
        />

        {shadeFaces3D}
        {shadingGroups}

        <polyline
          points={pts([A1, B1, C1, D1, A1])}
          fill="none"
          stroke="#94a3b8"
          strokeWidth="1.5"
        />

        <polyline
          points={seasonalSunPath("06-21")}
          fill="none"
          stroke={analysisDate === "06-21" ? "#111111" : "#2f2f2f"}
          strokeWidth={analysisDate === "06-21" ? "2.4" : "1.8"}
        />
        <polyline
          points={seasonalSunPath("03-21")}
          fill="none"
          stroke={analysisDate === "03-21" ? "#111111" : "#555555"}
          strokeWidth={analysisDate === "03-21" ? "2.2" : "1.8"}
          strokeDasharray="8 5"
        />
        <polyline
          points={seasonalSunPath("12-21")}
          fill="none"
          stroke={analysisDate === "12-21" ? "#111111" : "#2f2f2f"}
          strokeWidth={analysisDate === "12-21" ? "2.4" : "1.8"}
        />

        {timeMode !== "full_day" && (
          <circle cx={sunPt.x} cy={sunPt.y} r="3.5" fill="#111111" />
        )}

        <text x={juneLabelPt.x - 10} y={juneLabelPt.y - 12} fontSize="13" fill="#111111">
          June 21
        </text>
        <text x={marchLabelPt.x + 8} y={marchLabelPt.y - 8} fontSize="13" fill="#111111">
          March 21
        </text>
        <text x={decLabelPt.x + 8} y={decLabelPt.y + 5} fontSize="13" fill="#111111">
          Dec 21
        </text>

        <text x={northPt.x - 8} y={northPt.y - 8} fontSize="18" fill="#111111" fontWeight="500">
          N
        </text>
        <text x={eastPt.x + 8} y={eastPt.y + 4} fontSize="18" fill="#111111" fontWeight="500">
          E
        </text>
        <text x={southPt.x - 6} y={southPt.y + 18} fontSize="18" fill="#111111" fontWeight="500">
          S
        </text>
        <text x={westPt.x - 18} y={westPt.y + 4} fontSize="18" fill="#111111" fontWeight="500">
          W
        </text>

        <circle cx={sunPt.x} cy={sunPt.y} r="6" fill="#6b7280" />
        <text x={sunPt.x + 10} y={sunPt.y + 4} fontSize="11" fill="#6b7280">
          Sun
        </text>

        <text x="16" y="22" fontSize="12" fill="#334155">
          3D preview · {timeMode === "full_day" ? "all day overlay" : `${displayHour}:00`}
        </text>
      </svg>
    );
  }, [
  roomWidth,
  roomDepth,
  roomHeight,
  windowWidth,
  windowHeight,
  sillHeight,
  windowOffset,
  hasShading,
  shadingType,
  horizontalCount,
  verticalCount,
  horizontalDepth,
  verticalDepth,
  horizontalSpacing,
  verticalSpacing,
  shadingThickness,
  previewRotate,
  previewZoom,
  analysisDate,
  latitude,
  facadeAzimuth,
  displayHour,
  timeMode,
  daylightHours,
  orientation,
]);
  

  const showHorizontalControls =
    hasShading && (shadingType === "horizontal" || shadingType === "eggcrate");
  const showVerticalControls =
    hasShading && (shadingType === "vertical" || shadingType === "eggcrate");

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    dragRef.current.dragging = true;
    dragRef.current.lastX = e.clientX;
  }

  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
  if (!dragRef.current.dragging) return;
  const dx = e.clientX - dragRef.current.lastX;
  dragRef.current.lastX = e.clientX;
  setPreviewRotate((prev) => wrap360(prev + dx * 0.9));
}

  function onMouseUp() {
    dragRef.current.dragging = false;
  }

  function onWheel(e: React.WheelEvent<HTMLDivElement>) {
  e.preventDefault();
  const delta = e.deltaY > 0 ? -0.12 : 0.12;
  setPreviewZoom((prev) => Math.max(0.32, Math.min(2.8, prev + delta)));
}

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-7xl p-6">
        <h1 className="text-3xl font-bold tracking-tight">Glare Analysis</h1>
        <p className="mt-2 text-sm text-slate-600">
          Geometric direct sun patch study with configurable shading devices.
        </p>

             <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[380px_minmax(0,1fr)] items-start">
              <section className="self-start lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)]">
                <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border bg-white shadow-sm">
                  <div className="border-b px-4 py-3">
                    <h2 className="text-lg font-semibold">Inputs</h2>
                    <p className="mt-1 text-sm text-slate-500">
                      Adjust parameters while keeping the results visible.
                    </p>
                  </div>

                  <div className="flex-1 overflow-y-auto px-4 py-4 pb-8 space-y-6">
                    <div>
                      <h3 className="text-base font-semibold">Room</h3>
                      <NumberInput
                        label="Room Width (ft)"
                        value={roomWidth}
                        setValue={setRoomWidth}
                        step={1}
                        min={6}
                      />
                      <NumberInput
                        label="Room Depth (ft)"
                        value={roomDepth}
                        setValue={setRoomDepth}
                        step={1}
                        min={6}
                      />
                      <NumberInput
                        label="Room Height (ft)"
                        value={roomHeight}
                        setValue={setRoomHeight}
                        step={0.5}
                        min={8}
                      />
                    </div>

                    <div className="border-t pt-4">
                      <h3 className="text-base font-semibold">Window</h3>
                      <NumberInput
                        label="Window Width (ft)"
                        value={windowWidth}
                        setValue={setWindowWidth}
                        step={0.5}
                        min={2}
                      />
                      <NumberInput
                        label="Window Height (ft)"
                        value={windowHeight}
                        setValue={setWindowHeight}
                        step={0.5}
                        min={2}
                      />
                      <NumberInput
                        label="Sill Height (ft)"
                        value={sillHeight}
                        setValue={setSillHeight}
                        step={0.5}
                        min={0}
                      />
                      <NumberInput
                        label="Window Offset from Left (ft)"
                        value={windowOffset}
                        setValue={setWindowOffset}
                        step={0.5}
                        min={0}
                        max={Math.max(0, roomWidth - windowWidth)}
                      />
                    </div>

                    <div className="border-t pt-4">
                      <h3 className="text-base font-semibold">Sun / Site</h3>

                      <NumberInput
                        label="Latitude"
                        value={latitude}
                        setValue={setLatitude}
                        step={0.1}
                      />

                      <OrientationDial
                        value={orientationDeg}
                        onChange={setOrientationDeg}
                      />
                      <label className="mt-3 block">
                        <div className="mb-1 text-sm text-slate-600">Date</div>
                        <select
                          value={analysisDate}
                          onChange={(e) =>
                            setAnalysisDate(
                              e.target.value as "03-21" | "06-21" | "12-21"
                            )
                          }
                          className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
                        >
                          <option value="03-21">03-21</option>
                          <option value="06-21">06-21</option>
                          <option value="12-21">12-21</option>
                        </select>
                      </label>

                      <label className="mt-3 block">
                        <div className="mb-1 text-sm text-slate-600">Time</div>
                        <select
                          value={timeMode}
                          onChange={(e) =>
                            setTimeMode(
                              e.target.value as "full_day" | "9" | "12" | "15"
                            )
                          }
                          className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
                        >
                          <option value="full_day">All Day</option>
                          <option value="9">9:00</option>
                          <option value="12">12:00</option>
                          <option value="15">15:00</option>
                        </select>
                      </label>
                    </div>

                    <div className="border-t pt-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-base font-semibold">Shading</h3>
                        <label className="flex items-center gap-2 text-sm text-slate-600">
                          <input
                            type="checkbox"
                            checked={hasShading}
                            onChange={(e) => setHasShading(e.target.checked)}
                          />
                          Enable
                        </label>
                      </div>

                      {hasShading && (
                        <>
                          <label className="mt-3 block">
                            <div className="mb-1 text-sm text-slate-600">
                              Shading Type
                            </div>
                            <select
                              value={shadingType}
                              onChange={(e) =>
                                setShadingType(
                                  e.target.value as
                                    | "horizontal"
                                    | "vertical"
                                    | "eggcrate"
                                )
                              }
                              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
                            >
                              <option value="horizontal">Horizontal</option>
                              <option value="vertical">Vertical</option>
                              <option value="eggcrate">Eggcrate</option>
                            </select>
                          </label>

                          {(shadingType === "horizontal" ||
                            shadingType === "eggcrate") && (
                            <div className="mt-4 rounded-xl bg-slate-50 p-3">
                              <div className="text-sm font-medium text-slate-700">
                                Horizontal Fins
                              </div>
                              <NumberInput
                                label="Horizontal Count"
                                value={horizontalCount}
                                setValue={setHorizontalCount}
                                step={1}
                                min={1}
                              />
                              <NumberInput
                                label="Horizontal Depth (ft)"
                                value={horizontalDepth}
                                setValue={setHorizontalDepth}
                                step={0.25}
                                min={0.25}
                              />
                              <NumberInput
                                label="Horizontal Spacing (ft)"
                                value={horizontalSpacing}
                                setValue={setHorizontalSpacing}
                                step={0.25}
                                min={0.25}
                              />
                            </div>
                          )}

                          {(shadingType === "vertical" ||
                            shadingType === "eggcrate") && (
                            <div className="mt-4 rounded-xl bg-slate-50 p-3">
                              <div className="text-sm font-medium text-slate-700">
                                Vertical Fins
                              </div>
                              <NumberInput
                                label="Vertical Count"
                                value={verticalCount}
                                setValue={setVerticalCount}
                                step={1}
                                min={1}
                              />
                              <NumberInput
                                label="Vertical Depth (ft)"
                                value={verticalDepth}
                                setValue={setVerticalDepth}
                                step={0.25}
                                min={0.25}
                              />
                              <NumberInput
                                label="Vertical Spacing (ft)"
                                value={verticalSpacing}
                                setValue={setVerticalSpacing}
                                step={0.25}
                                min={0.25}
                              />
                            </div>
                          )}

                          <NumberInput
                            label="Shading Thickness (ft)"
                            value={shadingThickness}
                            setValue={setShadingThickness}
                            step={0.05}
                            min={0.05}
                          />
                        </>
                      )}
                    </div>

                    <div className="border-t pt-4">
                      <h3 className="text-base font-semibold">Analysis</h3>

                      <label className="mt-3 block">
                        <div className="mb-1 text-sm text-slate-600">
                          Scenario Name
                        </div>
                        <input
                          type="text"
                          value={scenarioName}
                          onChange={(e) => setScenarioName(e.target.value)}
                          className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
                          placeholder="Optional"
                        />
                      </label>

                      <div className="mt-4 flex flex-wrap gap-2">
                        <button
                          onClick={runAnalysis}
                          disabled={loading}
                          className="rounded-xl bg-slate-900 px-4 py-2 text-white hover:bg-slate-700 disabled:opacity-50"
                        >
                          {loading ? "Running..." : "Run Analysis"}
                        </button>

                        <button
                          onClick={saveCurrentScenario}
                          disabled={!result}
                          className="rounded-xl border px-4 py-2 hover:bg-slate-50 disabled:opacity-50"
                        >
                          Save Current Scenario
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              <section className="min-w-0 space-y-6 lg:h-[calc(100vh-3rem)] lg:overflow-y-auto lg:pr-2">
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() =>
                      exportSvgContainerAsPng(
                        frontPreviewRef.current,
                        "front-preview.png"
                      )
                    }
                    className="rounded-lg border px-3 py-2 text-sm hover:bg-slate-50"
                  >
                    Export Front
                  </button>

                  <button
                    onClick={() =>
                      exportSvgContainerAsPng(topPreviewRef.current, "top-preview.png")
                    }
                    className="rounded-lg border px-3 py-2 text-sm hover:bg-slate-50"
                  >
                    Export Top
                  </button>

                  <button
                    onClick={() =>
                      exportSvgContainerAsPng(
                        box3DPreviewRef.current,
                        "3d-preview.png"
                      )
                    }
                    className="rounded-lg border px-3 py-2 text-sm hover:bg-slate-50"
                  >
                    Export 3D
                  </button>

                  <button
                    onClick={() =>
                      exportElementAsPng(
                        comparisonRef.current,
                        "scenario-comparison.png"
                      )
                    }
                    className="rounded-lg border px-3 py-2 text-sm hover:bg-slate-50"
                  >
                    Export Comparison
                  </button>

                  <button
                    onClick={exportAllPreviews}
                    className="rounded-lg bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-700"
                  >
                    Export All
                  </button>
                </div>

                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div
                    ref={frontPreviewRef}
                    className="rounded-2xl border bg-white p-4 shadow-sm"
                  >
                    {frontPreview}
                  </div>

                  <div
                    ref={topPreviewRef}
                    className="rounded-2xl border bg-white p-4 shadow-sm"
                  >
                    {topPreview}
                  </div>
                </div>

                <div
                  ref={box3DPreviewRef}
                  className="rounded-2xl border bg-white p-4 shadow-sm select-none cursor-grab active:cursor-grabbing"
                  onMouseDown={onMouseDown}
                  onMouseMove={onMouseMove}
                  onMouseUp={onMouseUp}
                  onMouseLeave={onMouseUp}
                  onWheel={onWheel}
                >
                  <div className="mb-2 text-sm text-slate-500">
                    Drag to rotate · Scroll to zoom
                  </div>
                  {mounted ? box3DPreview : null}
                </div>

                {result && (
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                    <MetricCard
                      title="Best Time"
                      value={result.summary.best_time_label ?? "N/A"}
                    />
                    <MetricCard
                      title="Sun Patch %"
                      value={`${(
                        result.summary.average_coverage_ratio * 100
                      ).toFixed(1)}%`}
                    />
                    <MetricCard
                      title="Glare Area"
                      value={`${result.summary.max_sunlit_area_sqft.toFixed(1)} sf`}
                    />
                    <MetricCard
                      title="Max Penetration"
                      value={`${result.summary.max_penetration_ft.toFixed(1)} ft`}
                    />
                  </div>
                )}

                {result && (
                  <div className="overflow-hidden rounded-2xl border bg-white shadow-sm">
                    <div className="border-b bg-slate-50 px-4 py-3">
                      <h3 className="text-lg font-semibold">Hourly Results</h3>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead className="bg-slate-50 text-slate-600">
                          <tr>
                            <th className="px-4 py-3 text-left font-medium">Time</th>
                            <th className="px-4 py-3 text-left font-medium">
                              Sun Visible
                            </th>
                            <th className="px-4 py-3 text-left font-medium">
                              Altitude
                            </th>
                            <th className="px-4 py-3 text-left font-medium">
                              Azimuth
                            </th>
                            <th className="px-4 py-3 text-left font-medium">
                              Glare Area
                            </th>
                            <th className="px-4 py-3 text-left font-medium">
                              Coverage
                            </th>
                            <th className="px-4 py-3 text-left font-medium">
                              Max Penetration
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.times.map((t) => (
                            <tr key={t.label} className="border-t">
                              <td className="px-4 py-3">{t.label}</td>
                              <td className="px-4 py-3">
                                {t.sun_visible ? "Yes" : "No"}
                              </td>
                              <td className="px-4 py-3">
                                {t.altitude_deg == null
                                  ? "—"
                                  : `${t.altitude_deg.toFixed(1)}°`}
                              </td>
                              <td className="px-4 py-3">
                                {t.azimuth_deg == null
                                  ? "—"
                                  : `${t.azimuth_deg.toFixed(1)}°`}
                              </td>
                              <td className="px-4 py-3">
                                {t.sunlit_area_sqft.toFixed(2)} sf
                              </td>
                              <td className="px-4 py-3">
                                {(t.coverage_ratio * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-3">
                                {t.max_penetration_ft.toFixed(2)} ft
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {savedScenarios.length > 0 && (
                  <div
                    id="saved-scenarios"
                    className="rounded-2xl border bg-white p-4 shadow-sm"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold">Saved Scenarios</h3>
                      <span className="text-sm text-slate-500">
                        Select cards to compare
                      </span>
                    </div>

                    <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
                      {savedScenarios.map((scenario) => {
                        const isSelected = selectedScenarioIds.includes(scenario.id);

                        return (
                          <div
                            key={scenario.id}
                            className={`rounded-xl border p-3 transition ${
                              isSelected
                                ? "border-slate-900 bg-slate-50"
                                : "border-slate-200 bg-white"
                            }`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <label className="flex items-start gap-2 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  onChange={() =>
                                    toggleScenarioSelection(scenario.id)
                                  }
                                  className="mt-1"
                                />
                                <div>
                                  <div className="font-semibold text-slate-900">
                                    {scenario.name}
                                  </div>
                                  <div className="mt-1 text-xs text-slate-500">
                                    {scenario.inputs.hasShading
                                      ? `${scenario.inputs.shadingType} shading`
                                      : "no shading"}
                                  </div>
                                </div>
                              </label>

                              <button
                                onClick={() => deleteScenario(scenario.id)}
                                className="text-xs text-slate-500 hover:underline"
                              >
                                Delete
                              </button>
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                              <div>
                                <div className="text-slate-500">Orientation</div>
                                <div className="font-medium">
                                  {scenario.inputs.orientation}
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-500">Date</div>
                                <div className="font-medium">
                                  {scenario.inputs.analysisDate}
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-500">Time</div>
                                <div className="font-medium">
                                  {scenario.inputs.timeMode === "full_day"
                                    ? "All Day"
                                    : `${scenario.inputs.timeMode}:00`}
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-500">Window</div>
                                <div className="font-medium">
                                  {scenario.inputs.windowWidth}w ×{" "}
                                  {scenario.inputs.windowHeight}h
                                </div>
                              </div>
                            </div>

                            {scenario.inputs.hasShading && (
                              <div className="mt-3 rounded-lg bg-slate-50 p-2 text-xs text-slate-600">
                                {scenario.inputs.shadingType === "horizontal" && (
                                  <>
                                    <div>Horizontal fins</div>
                                    <div>
                                      {scenario.inputs.horizontalCount} fins · depth{" "}
                                      {scenario.inputs.horizontalDepth} ft · spacing{" "}
                                      {scenario.inputs.horizontalSpacing} ft
                                    </div>
                                  </>
                                )}

                                {scenario.inputs.shadingType === "vertical" && (
                                  <>
                                    <div>Vertical fins</div>
                                    <div>
                                      {scenario.inputs.verticalCount} fins · depth{" "}
                                      {scenario.inputs.verticalDepth} ft · spacing{" "}
                                      {scenario.inputs.verticalSpacing} ft
                                    </div>
                                  </>
                                )}

                                {scenario.inputs.shadingType === "eggcrate" && (
                                  <>
                                    <div>Eggcrate shading</div>
                                    <div>
                                      H: {scenario.inputs.horizontalCount} /{" "}
                                      {scenario.inputs.horizontalDepth} ft {" · "}V:{" "}
                                      {scenario.inputs.verticalCount} /{" "}
                                      {scenario.inputs.verticalDepth} ft
                                    </div>
                                  </>
                                )}
                              </div>
                            )}

                            <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 border-t pt-3 text-sm">
                              <div>
                                <div className="text-slate-500">Best Time</div>
                                <div className="font-medium">
                                  {scenario.result.summary.best_time_label ?? "N/A"}
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-500">Glare Area</div>
                                <div className="font-medium">
                                  {scenario.result.summary.max_sunlit_area_sqft.toFixed(
                                    1
                                  )}{" "}
                                  sf
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-500">Coverage</div>
                                <div className="font-medium">
                                  {(
                                    scenario.result.summary.average_coverage_ratio *
                                    100
                                  ).toFixed(1)}
                                  %
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-500">Max Penetration</div>
                                <div className="font-medium">
                                  {scenario.result.summary.max_penetration_ft.toFixed(
                                    1
                                  )}{" "}
                                  ft
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {selectedScenarios.length >= 2 && (
                  <div
                    ref={comparisonRef}
                    className="overflow-hidden rounded-2xl border bg-white shadow-sm"
                  >
                    <div className="border-b bg-slate-50 px-4 py-3">
                      <div className="flex items-center justify-between gap-4">
                        <h3 className="text-lg font-semibold">
                          Scenario Comparison
                        </h3>

                        <label className="block">
                          <div className="mb-1 text-xs text-slate-500">Metric</div>
                          <select
                            value={comparisonMetric}
                            onChange={(e) =>
                              setComparisonMetric(
                                e.target.value as
                                  | "sunlit_area_sqft"
                                  | "coverage_ratio"
                                  | "max_penetration_ft"
                              )
                            }
                            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
                          >
                            <option value="sunlit_area_sqft">Glare Area</option>
                            <option value="coverage_ratio">Coverage %</option>
                            <option value="max_penetration_ft">Max Penetration</option>
                          </select>
                        </label>
                      </div>
                    </div>

                    <div className="p-4">
                      <ScenarioLineChart
                        scenarios={selectedScenarios}
                        metric={comparisonMetric}
                      />
                    </div>

                    <div className="border-t">
                      <div className="overflow-x-auto">
                        <table className="min-w-full text-sm">
                          <thead className="bg-slate-50 text-slate-600">
                            <tr>
                              <th className="px-4 py-3 text-left font-medium">
                                Scenario
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Shading
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Orientation
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Date
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Time
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Best Time
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Glare Area
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Avg Coverage
                              </th>
                              <th className="px-4 py-3 text-left font-medium">
                                Max Penetration
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedScenarios.map((scenario) => (
                              <tr key={scenario.id} className="border-t">
                                <td className="px-4 py-3 font-medium">
                                  {scenario.name}
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.inputs.hasShading
                                    ? scenario.inputs.shadingType
                                    : "none"}
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.inputs.orientation}
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.inputs.analysisDate}
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.inputs.timeMode === "full_day"
                                    ? "All Day"
                                    : `${scenario.inputs.timeMode}:00`}
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.result.summary.best_time_label ?? "N/A"}
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.result.summary.max_sunlit_area_sqft.toFixed(
                                    2
                                  )}{" "}
                                  sf
                                </td>
                                <td className="px-4 py-3">
                                  {(
                                    scenario.result.summary.average_coverage_ratio *
                                    100
                                  ).toFixed(1)}
                                  %
                                </td>
                                <td className="px-4 py-3">
                                  {scenario.result.summary.max_penetration_ft.toFixed(
                                    2
                                  )}{" "}
                                  ft
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="rounded-2xl border border-red-300 bg-red-50 p-4 text-red-700">
                    {error}
                  </div>
                )}
              </section>
            </div>
          </div>
        </main>
  );
}
