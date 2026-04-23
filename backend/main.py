from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)
import math

# Honeybee Core
from honeybee.model import Model
from honeybee.room import Room
from honeybee.aperture import Aperture
from honeybee.shade import Shade
from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybug_geometry.geometry3d.face import Face3D



# =============================
# basic helpers
# =============================
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def resolve_date(analysis_date: str):
    mapping = {
        "03-21": (3, 21),
        "06-21": (6, 21),
        "12-21": (12, 21),
    }
    return mapping.get(analysis_date, (6, 21))





def day_of_year(month: int, day: int) -> int:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = clamp(month, 1, 12)
    day = clamp(day, 1, month_days[month - 1])
    return sum(month_days[: month - 1]) + day


def solar_position(latitude_deg: float, month: int, day: int, hour_local: float):
    n = day_of_year(month, day)
    lat = math.radians(latitude_deg)

    gamma = 2.0 * math.pi / 365.0 * (n - 1 + (hour_local - 12.0) / 24.0)

    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    hra_deg = 15.0 * (hour_local - 12.0)
    hra = math.radians(hra_deg)

    sin_alt = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hra)
    sin_alt = clamp(sin_alt, -1.0, 1.0)
    alt = math.asin(sin_alt)

    cos_az = (math.sin(decl) - math.sin(alt) * math.sin(lat)) / (math.cos(alt) * math.cos(lat) + 1e-9)
    cos_az = clamp(cos_az, -1.0, 1.0)
    az = math.degrees(math.acos(cos_az))

    if hra > 0:
        az = 360.0 - az

    return math.degrees(alt), az


def sun_vector_world(altitude_deg: float, azimuth_deg: float):
    alt = math.radians(altitude_deg)
    az = math.radians(azimuth_deg)
    x = math.sin(az) * math.cos(alt)
    y = math.cos(az) * math.cos(alt)
    z = math.sin(alt)
    return (x, y, z)


def world_to_room_local(vec_world, facade_azimuth_deg):
    wx, wy, wz = vec_world
    a = math.radians(facade_azimuth_deg)

    outward = (math.sin(a), math.cos(a), 0.0)
    inward = (-outward[0], -outward[1], 0.0)
    right = (math.cos(a), -math.sin(a), 0.0)

    lx = wx * right[0] + wy * right[1]
    ly = wx * inward[0] + wy * inward[1]
    lz = wz

    return (lx, ly, lz)


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# =============================
# ray-face intersections
# =============================
def point_in_triangle_3d(p, a, b, c, tol=1e-7):
    v0 = vec_sub(c, a)
    v1 = vec_sub(b, a)
    v2 = vec_sub(p, a)

    dot00 = vec_dot(v0, v0)
    dot01 = vec_dot(v0, v1)
    dot02 = vec_dot(v0, v2)
    dot11 = vec_dot(v1, v1)
    dot12 = vec_dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < tol:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv

    return u >= -tol and v >= -tol and (u + v) <= 1.0 + tol


def point_in_quad_3d(p, quad_pts, tol=1e-7):
    a, b, c, d = quad_pts
    return point_in_triangle_3d(p, a, b, c, tol) or point_in_triangle_3d(p, a, c, d, tol)


def ray_face_intersection(origin, direction, quad_pts, tol=1e-7):
    a, b, c, d = quad_pts

    ab = vec_sub(b, a)
    ad = vec_sub(d, a)
    normal = vec_cross(ab, ad)

    denom = vec_dot(normal, direction)
    if abs(denom) < tol:
        return False

    t = vec_dot(normal, vec_sub(a, origin)) / denom
    if t <= tol:
        return False

    hit = vec_add(origin, vec_scale(direction, t))
    return point_in_quad_3d(hit, quad_pts, tol)


def line_hits_any_face(origin, direction, faces):
    for face in faces:
        if ray_face_intersection(origin, direction, face["pts"]):
            return True
    return False


# =============================
# Honeybee geometry builders
# =============================
def get_front_face(room):
    return min(room.faces, key=lambda f: f.geometry.center.y)


def create_aperture_geometry(room_width, window_width, window_height, sill_height, window_offset):
    x0 = max(0.0, min(window_offset, room_width - window_width))
    x1 = x0 + window_width
    z0 = sill_height
    z1 = sill_height + window_height

    pts = [
        Point3D(x0, 0.0, z0),
        Point3D(x1, 0.0, z0),
        Point3D(x1, 0.0, z1),
        Point3D(x0, 0.0, z1),
    ]
    return Face3D(pts)


def create_horizontal_hb_shades(
    room_width,
    window_width,
    window_height,
    sill_height,
    window_offset,
    count,
    depth,
    spacing,
    thickness,
):
    shades = []

    window_left = max(0.0, min(window_offset, room_width - window_width))
    window_right = window_left + window_width
    head = sill_height + window_height

    for i in range(int(count)):
        z_top = head - i * spacing
        z_bottom = z_top - thickness

        if z_bottom < sill_height:
            continue

        pts = [
            Point3D(window_left, -depth, z_bottom),
            Point3D(window_right, -depth, z_bottom),
            Point3D(window_right, 0.0, z_top),
            Point3D(window_left, 0.0, z_top),
        ]
        shades.append(Shade(f"h_shade_{i}", Face3D(pts)))

    return shades


def create_vertical_hb_shades(
    room_width,
    window_width,
    window_height,
    sill_height,
    window_offset,
    count,
    depth,
    spacing,
    thickness,
):
    shades = []

    window_left = max(0.0, min(window_offset, room_width - window_width))
    window_right = window_left + window_width

    for i in range(int(count)):
        x0 = window_left + i * spacing
        if x0 >= window_right:
            continue

        x1 = min(x0 + thickness, window_right)
        if x1 <= x0:
            continue

        pts = [
            Point3D(x0, -depth, sill_height),
            Point3D(x1, -depth, sill_height),
            Point3D(x1, 0.0, sill_height + window_height),
            Point3D(x0, 0.0, sill_height + window_height),
        ]
        shades.append(Shade(f"v_shade_{i}", Face3D(pts)))

    return shades


def build_honeybee_model(data):
    room_width = float(data.get("roomWidth", 20.0))
    room_depth = float(data.get("roomDepth", 20.0))
    room_height = float(data.get("roomHeight", 10.0))
    window_width = float(data.get("windowWidth", 8.0))
    window_height = float(data.get("windowHeight", 6.0))
    sill_height = float(data.get("sillHeight", 3.0))
    window_offset = float(
        data.get("windowOffset", max(0.0, (room_width - window_width) / 2.0))
    )

    shading_type = data.get("shadingType", "horizontal")
    has_shading = bool(data.get("hasShading", False))

    h_count = int(data.get("horizontalCount", 3))
    v_count = int(data.get("verticalCount", 3))
    h_depth = float(data.get("horizontalDepth", 2.0))
    v_depth = float(data.get("verticalDepth", 2.0))
    h_spacing = float(data.get("horizontalSpacing", 1.5))
    v_spacing = float(data.get("verticalSpacing", 1.5))
    thickness = float(data.get("shadingThickness", 0.25))

    room = Room.from_box("TestRoom", room_width, room_depth, room_height)

    front_face = get_front_face(room)
    aperture_geo = create_aperture_geometry(
        room_width=room_width,
        window_width=window_width,
        window_height=window_height,
        sill_height=sill_height,
        window_offset=window_offset,
    )
    aperture = Aperture("MainWindow", aperture_geo)
    front_face.add_aperture(aperture)

    shades = []
    if has_shading:
        if shading_type in ["horizontal", "eggcrate"]:
            shades.extend(
                create_horizontal_hb_shades(
                    room_width=room_width,
                    window_width=window_width,
                    window_height=window_height,
                    sill_height=sill_height,
                    window_offset=window_offset,
                    count=h_count,
                    depth=h_depth,
                    spacing=h_spacing,
                    thickness=thickness,
                )
            )

        if shading_type in ["vertical", "eggcrate"]:
            shades.extend(
                create_vertical_hb_shades(
                    room_width=room_width,
                    window_width=window_width,
                    window_height=window_height,
                    sill_height=sill_height,
                    window_offset=window_offset,
                    count=v_count,
                    depth=v_depth,
                    spacing=v_spacing,
                    thickness=thickness,
                )
            )

    for shd in shades:
        aperture.add_outdoor_shade(shd)

    model = Model("GlareModel", [room])

    validation = {
        "room_check": "ok",
        "model_check": "ok",
    }

    try:
        room.check_planar(raise_exception=True)
    except Exception as e:
        validation["room_check"] = str(e)

    try:
        model.check_all()
    except Exception as e:
        validation["model_check"] = str(e)

    return model, {
        "hb_room_identifier": room.identifier,
        "hb_aperture_identifier": aperture.identifier,
        "hb_shade_count": len(shades),
        "hb_validation": validation,
    }


# =============================
# shading faces for analysis
# =============================
def generate_shading_faces(data):
    if not data.get("hasShading", False):
        return []

    room_width = float(data.get("roomWidth", 20))
    window_width = float(data.get("windowWidth", 8))
    window_height = float(data.get("windowHeight", 6))
    sill_height = float(data.get("sillHeight", 3))

    shading_type = data.get("shadingType", "horizontal")

    h_count = int(data.get("horizontalCount", 3))
    v_count = int(data.get("verticalCount", 3))

    h_depth = float(data.get("horizontalDepth", 2))
    v_depth = float(data.get("verticalDepth", 2))

    h_spacing = float(data.get("horizontalSpacing", 1.5))
    v_spacing = float(data.get("verticalSpacing", 1.5))

    thickness = float(data.get("shadingThickness", 0.25))

    window_offset = float(
        data.get("windowOffset", max(0.0, (room_width - window_width) / 2.0))
    )
    window_left = max(0.0, min(window_offset, room_width - window_width))
    window_right = window_left + window_width

    faces = []

    if shading_type in ["horizontal", "eggcrate"]:
        head = sill_height + window_height
        for i in range(h_count):
            z_top = head - i * h_spacing
            z_bottom = z_top - thickness
            if z_bottom < sill_height:
                continue

            pts = [
                (window_left, -h_depth, z_bottom),
                (window_right, -h_depth, z_bottom),
                (window_right, 0.0, z_top),
                (window_left, 0.0, z_top),
            ]
            faces.append({"pts": pts, "type": "horizontal", "index": i})

    if shading_type in ["vertical", "eggcrate"]:
        for i in range(v_count):
            x0 = window_left + i * v_spacing
            if x0 >= window_right:
                continue

            x1 = min(x0 + thickness, window_right)
            if x1 <= x0:
                continue

            pts = [
                (x0, -v_depth, sill_height),
                (x1, -v_depth, sill_height),
                (x1, 0.0, sill_height + window_height),
                (x0, 0.0, sill_height + window_height),
            ]
            faces.append({"pts": pts, "type": "vertical", "index": i})

    return faces


# =============================
# analysis helpers
# =============================
def get_time_samples(time_mode: str):
    if time_mode == "9":
        return [9.0]
    if time_mode == "12":
        return [12.0]
    if time_mode == "15":
        return [15.0]
    return [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


def floor_grid(room_width, room_depth, grid_size=0.25):
    points = []
    x = grid_size / 2.0
    while x < room_width:
        y = grid_size / 2.0
        while y < room_depth:
            points.append((x, y, 0.01))
            y += grid_size
        x += grid_size
    return points


def effective_max_penetration_from_lit_points(
    lit_points,
    grid_size,
    window_width,
    min_width_ratio=0.08,
    min_cells_floor=1,
):
    if not lit_points:
        return 0.0

    rows = {}
    for x, y, _ in lit_points:
        row_idx = int(math.floor(y / grid_size))
        col_idx = int(math.floor(x / grid_size))
        rows.setdefault(row_idx, set()).add(col_idx)

    min_required_width = max(grid_size * min_cells_floor, window_width * min_width_ratio)

    farthest_valid_y = 0.0

    for row_idx, cols in rows.items():
        lit_width = len(cols) * grid_size
        if lit_width + 1e-9 >= min_required_width:
            y_val = (row_idx + 1) * grid_size
            farthest_valid_y = max(farthest_valid_y, y_val)

    return farthest_valid_y


def point_sees_sun_through_window(
    point,
    sun_dir_local,
    window_left,
    window_right,
    sill_height,
    window_height,
    shading_faces
):
    if sun_dir_local[1] <= 1e-9:
        return False

    to_sun = (-sun_dir_local[0], -sun_dir_local[1], -sun_dir_local[2])

    if abs(to_sun[1]) < 1e-9:
        return False

    t_plane = (0.0 - point[1]) / to_sun[1]
    if t_plane <= 0:
        return False

    hit = vec_add(point, vec_scale(to_sun, t_plane))
    hx, _, hz = hit

    if not (window_left <= hx <= window_right):
        return False
    if not (sill_height <= hz <= sill_height + window_height):
        return False

    origin_outside = vec_add(hit, vec_scale(to_sun, 1e-4))
    if line_hits_any_face(origin_outside, to_sun, shading_faces):
        return False

    return True


def analyze_single_time(data, hour_local, shading_faces):
    room_width = float(data.get("roomWidth", 20.0))
    room_depth = float(data.get("roomDepth", 20.0))
    window_width = float(data.get("windowWidth", 8.0))
    window_height = float(data.get("windowHeight", 6.0))
    sill_height = float(data.get("sillHeight", 3.0))
    window_offset = float(
        data.get("windowOffset", max(0.0, (room_width - window_width) / 2.0))
    )
    latitude = float(data.get("latitude", 47.61))
    analysis_date = data.get("analysisDate", "06-21")

    month, day = resolve_date(analysis_date)
    facade_azimuth = float(data.get("orientationDeg", 180.0))

    window_left = max(0.0, min(window_offset, room_width - window_width))
    window_right = window_left + window_width

    altitude_deg, azimuth_deg = solar_position(latitude, month, day, hour_local)

    if altitude_deg <= 0:
        return {
            "label": f"{int(hour_local):02d}:00",
            "hour": hour_local,
            "sun_visible": False,
            "altitude_deg": None,
            "azimuth_deg": None,
            "sunlit_area_sqft": 0.0,
            "coverage_ratio": 0.0,
            "max_penetration_ft": 0.0,
        }

    sun_world = sun_vector_world(altitude_deg, azimuth_deg)
    travel_world = (-sun_world[0], -sun_world[1], -sun_world[2])
    sun_dir_local = world_to_room_local(travel_world, facade_azimuth)

    if sun_dir_local[1] <= 1e-9:
        return {
            "label": f"{int(hour_local):02d}:00",
            "hour": hour_local,
            "sun_visible": False,
            "altitude_deg": altitude_deg,
            "azimuth_deg": azimuth_deg,
            "sunlit_area_sqft": 0.0,
            "coverage_ratio": 0.0,
            "max_penetration_ft": 0.0,
        }

    grid_size = float(data.get("gridSize", 0.25))
    points = floor_grid(room_width, room_depth, grid_size)
    cell_area = grid_size * grid_size

    lit_points = []

    for p in points:
        visible = point_sees_sun_through_window(
            point=p,
            sun_dir_local=sun_dir_local,
            window_left=window_left,
            window_right=window_right,
            sill_height=sill_height,
            window_height=window_height,
            shading_faces=shading_faces,
        )

        if visible:
            lit_points.append(p)

    sunlit_area = len(lit_points) * cell_area
    total_floor_area = room_width * room_depth
    coverage_ratio = sunlit_area / total_floor_area if total_floor_area > 0 else 0.0

    max_penetration = effective_max_penetration_from_lit_points(
        lit_points=lit_points,
        grid_size=grid_size,
        window_width=window_width,
        min_width_ratio=0.08,
        min_cells_floor=1,
    )

    return {
        "label": f"{int(hour_local):02d}:00",
        "hour": hour_local,
        "sun_visible": True,
        "altitude_deg": altitude_deg,
        "azimuth_deg": azimuth_deg,
        "sunlit_area_sqft": sunlit_area,
        "coverage_ratio": coverage_ratio,
        "max_penetration_ft": max_penetration,
    }


# =============================
# routes
# =============================
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.get_json(force=True)

        room_width = float(data.get("roomWidth", 20.0))
        room_depth = float(data.get("roomDepth", 20.0))
        room_height = float(data.get("roomHeight", 10.0))
        window_width = float(data.get("windowWidth", 8.0))
        window_height = float(data.get("windowHeight", 6.0))
        sill_height = float(data.get("sillHeight", 3.0))
        window_offset = float(
            data.get("windowOffset", max(0.0, (room_width - window_width) / 2.0))
        )

        if room_width <= 0 or room_depth <= 0 or room_height <= 0:
            return jsonify({"ok": False, "error": "Room dimensions must be positive."}), 400

        if window_width <= 0 or window_height <= 0:
            return jsonify({"ok": False, "error": "Window dimensions must be positive."}), 400

        if window_width > room_width:
            return jsonify({"ok": False, "error": "Window width cannot exceed room width."}), 400

        if window_offset < 0 or window_offset > max(0.0, room_width - window_width):
            return jsonify({"ok": False, "error": "Window offset is out of valid range."}), 400

        if sill_height < 0:
            return jsonify({"ok": False, "error": "Sill height cannot be negative."}), 400

        if sill_height + window_height > room_height:
            return jsonify({"ok": False, "error": "Window top cannot exceed room height."}), 400

        hb_model, hb_debug = build_honeybee_model(data)

        shading_faces = generate_shading_faces(data)
        time_mode = data.get("timeMode", "full_day")
        times = get_time_samples(time_mode)

        results = [analyze_single_time(data, t, shading_faces) for t in times]

        valid = [r for r in results if r["sun_visible"]]
        if valid:
            best = max(valid, key=lambda r: r["sunlit_area_sqft"])
            avg_cov = sum(r["coverage_ratio"] for r in valid) / len(valid)
            max_pen = max(r["max_penetration_ft"] for r in valid)
            max_area = max(r["sunlit_area_sqft"] for r in valid)
            best_label = best["label"]
        else:
            avg_cov = 0.0
            max_pen = 0.0
            max_area = 0.0
            best_label = None

        return jsonify({
            "ok": True,
            "summary": {
                "best_time_label": best_label,
                "max_sunlit_area_sqft": max_area,
                "average_coverage_ratio": avg_cov,
                "max_penetration_ft": max_pen,
            },
            "times": results,
            "debug": {
                "shading_count_generated": len(shading_faces),
                "hb_shade_count": hb_debug["hb_shade_count"],
                "hb_room_identifier": hb_debug["hb_room_identifier"],
                "hb_aperture_identifier": hb_debug["hb_aperture_identifier"],
                "hb_validation": hb_debug["hb_validation"],
            },
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"ok": True, "message": "Backend running with face-based shading intersection"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)