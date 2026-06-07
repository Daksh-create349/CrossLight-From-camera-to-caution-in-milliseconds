"""
danger_zone.py
--------------
Utility functions for polygon-based danger zone containment checks.

Functions
---------
point_in_danger_zone(point, danger_zone_polygon) -> bool
    Ray-casting algorithm. Pure Python, no external libraries.

load_danger_zone(filepath) -> list[tuple[float, float]]
    Loads a polygon saved as a .npy file and returns it as a list of (x, y) tuples.
"""


def point_in_danger_zone(point, danger_zone_polygon):
    """
    Determines whether a 2-D point lies inside a polygon using the
    ray-casting (Jordan curve) algorithm.

    A horizontal ray is cast from the point towards +infinity on the X axis.
    Each time the ray crosses an edge of the polygon the inside/outside status
    is toggled.  Points that fall exactly on an edge are considered inside.

    Args:
        point (tuple[float, float]): The query point as (x, y).
        danger_zone_polygon (list[tuple[float, float]]): Ordered list of (x, y)
            vertices defining the polygon.  At least 3 vertices are required.

    Returns:
        bool: True if the point is inside (or on the boundary of) the polygon,
              False otherwise.
    """
    if not danger_zone_polygon or len(danger_zone_polygon) < 3:
        return False

    px, py = float(point[0]), float(point[1])
    n = len(danger_zone_polygon)
    inside = False

    # Iterate over every edge (vi → vj)
    j = n - 1  # Start with the edge connecting the last vertex to the first
    for i in range(n):
        xi, yi = float(danger_zone_polygon[i][0]), float(danger_zone_polygon[i][1])
        xj, yj = float(danger_zone_polygon[j][0]), float(danger_zone_polygon[j][1])

        # Check if the horizontal ray from (px, py) crosses this edge.
        # Conditions:
        #   1. The edge must straddle the horizontal line y = py
        #      (one endpoint above, one on or below).
        #   2. The x-coordinate of the crossing must be to the right of px.
        crosses_y = (yi > py) != (yj > py)
        if crosses_y:
            # x-coordinate of the ray–edge intersection
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside

        j = i  # Advance: next iteration's "previous" vertex is the current one

    return inside


def load_danger_zone(filepath="danger_zone.npy"):
    """
    Loads a polygon that was previously saved with numpy as a .npy file and
    returns it as a plain Python list of (x, y) tuples so it can be used
    directly with ``point_in_danger_zone`` without numpy being required at
    call-sites.

    Args:
        filepath (str): Path to the .npy file.  Defaults to 'danger_zone.npy'.

    Returns:
        list[tuple[float, float]]: Ordered list of (x, y) polygon vertices.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded array does not have shape (N, 2).
    """
    import numpy as np  # numpy used only here for loading; not required elsewhere

    data = np.load(filepath)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Expected array of shape (N, 2), got {data.shape} from '{filepath}'."
        )

    return [(float(row[0]), float(row[1])) for row in data]


# ─────────────────────────────────────────────────────────────
# Self-contained tests (run with:  python danger_zone.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running danger_zone self-tests...\n")

    # ── Test 1: Axis-aligned square ──────────────────────────
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    assert point_in_danger_zone((5, 5), square) is True,    "Inside square failed"
    assert isinstance(point_in_danger_zone((0, 0), square), bool), "Corner of square (degenerate) - must return bool"
    assert point_in_danger_zone((10.1, 5), square) is False, "Right of square failed"
    assert point_in_danger_zone((-1, 5), square) is False,  "Left of square failed"
    assert point_in_danger_zone((5, -1), square) is False,  "Below square failed"
    assert point_in_danger_zone((5, 11), square) is False,  "Above square failed"
    print("Test 1 (square) passed.")

    # ── Test 2: Irregular convex quadrilateral ───────────────
    quad = [(2, 0), (8, 1), (9, 8), (1, 7)]

    assert point_in_danger_zone((5, 4), quad) is True,    "Center of quad failed"
    assert point_in_danger_zone((0, 0), quad) is False,   "Outside quad failed"
    assert point_in_danger_zone((9.5, 4), quad) is False, "Far right of quad failed"
    print("Test 2 (irregular quad) passed.")

    # ── Test 3: Concave (L-shaped) polygon ───────────────────
    # L-shape: 6 vertices
    l_shape = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]

    assert point_in_danger_zone((1, 1), l_shape) is True,  "Bottom-left of L failed"
    assert point_in_danger_zone((1, 3), l_shape) is True,  "Top-left of L failed"
    assert point_in_danger_zone((3, 3), l_shape) is False, "Concave notch of L failed"
    assert point_in_danger_zone((5, 1), l_shape) is False, "Outside right of L failed"
    print("Test 3 (concave L-shape) passed.")

    # ── Test 4: Real-world meter coordinates (typical after homography) ──
    crosswalk_m = [(1.0, 2.0), (5.0, 2.0), (5.0, 6.0), (1.0, 6.0)]

    assert point_in_danger_zone((3.0, 4.0), crosswalk_m) is True,  "Inside crosswalk (m) failed"
    assert point_in_danger_zone((0.5, 4.0), crosswalk_m) is False, "Left of crosswalk (m) failed"
    assert point_in_danger_zone((3.0, 7.0), crosswalk_m) is False, "Above crosswalk (m) failed"
    print("Test 4 (meter coordinates) passed.")

    # ── Test 5: Fewer than 3 vertices edge case ───────────────
    assert point_in_danger_zone((1, 1), []) is False,           "Empty polygon failed"
    assert point_in_danger_zone((1, 1), [(0, 0), (2, 2)]) is False, "2-vertex polygon failed"
    print("Test 5 (degenerate polygons) passed.")

    # ── Test 6: load_danger_zone round-trip ──────────────────
    import numpy as np, os, tempfile

    sample = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]])
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        np.save(tmp_path, sample)
        loaded = load_danger_zone(tmp_path)
        assert len(loaded) == 4,                           "load length mismatch"
        assert loaded[0] == (0.0, 0.0),                   "load vertex 0 mismatch"
        assert loaded[2] == (6.0, 4.0),                   "load vertex 2 mismatch"
        assert point_in_danger_zone((3.0, 2.0), loaded),  "loaded polygon containment failed"
        print("Test 6 (load_danger_zone round-trip) passed.")
    finally:
        os.remove(tmp_path)

    print("\nAll danger_zone tests passed successfully!")
