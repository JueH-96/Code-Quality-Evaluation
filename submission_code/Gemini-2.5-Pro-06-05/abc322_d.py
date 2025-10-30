import sys

def solve():
    """
    Solves the polyomino tiling problem by generating all valid placements
    and using a backtracking search to find a perfect tiling.
    """

    # Helper functions are nested for encapsulation.
    def read_grid():
        """Reads a 4x4 grid from stdin, handling potential EOF."""
        try:
            grid = [sys.stdin.readline().strip() for _ in range(4)]
            # Check if the read was successful (not empty due to EOF)
            if all(grid) and all(len(row) == 4 for row in grid):
                return grid
            return None
        except (IOError, IndexError):
            return None

    def get_coords(grid):
        """Extracts '#' coordinates from a grid representation."""
        coords = set()
        for r, row in enumerate(grid):
            for c, char in enumerate(row):
                if char == '#':
                    coords.add((r, c))
        return coords

    def normalize(coords):
        """Translates a shape so its top-leftmost point is at (0,0)."""
        if not coords:
            return frozenset()
        min_r = min(r for r, c in coords)
        min_c = min(c for r, c in coords)
        return frozenset((r - min_r, c - min_c) for r, c in coords)

    def rotate(coords):
        """
        Rotates a normalized shape 90 degrees clockwise. This transformation
        preserves chirality (i.e., it does not flip the piece).
        A 90-degree clockwise rotation of a point (x, y) is (y, -x).
        We map grid coordinates (c, r) to (x, y), so (c,r) becomes (r,-c).
        """
        rotated_raw = set((c, -r) for r, c in coords)
        return normalize(rotated_raw)

    # --- Main logic of the solver ---

    # 1. Read input grids
    poly_grids = [read_grid() for _ in range(3)]
    if any(p is None for p in poly_grids):
        print("No")
        return

    # 2. Extract coordinates and check total cell count
    poly_coords_list = [get_coords(grid) for grid in poly_grids]
    if sum(len(p) for p in poly_coords_list) != 16:
        print("No")
        return

    # 3. Generate all unique rotated shapes for each polyomino
    all_poly_shapes = []
    for p_coords in poly_coords_list:
        unique_shapes = set()
        current_shape = normalize(p_coords)
        for _ in range(4):
            unique_shapes.add(current_shape)
            current_shape = rotate(current_shape)
        all_poly_shapes.append(list(unique_shapes))

    # 4. Generate all possible valid placements on the 4x4 grid
    all_poly_placements = []
    for shapes in all_poly_shapes:
        placements = set()
        for shape in shapes:
            if not shape:
                continue
            
            # Get the bounding box of the normalized shape
            max_r = max(r for r, c in shape) if shape else -1
            max_c = max(c for r, c in shape) if shape else -1
            
            # Iterate through all valid top-left anchor points (offsets)
            for r_offset in range(4 - max_r):
                for c_offset in range(4 - max_c):
                    placement = frozenset((r + r_offset, c + c_offset) for r, c in shape)
                    placements.add(placement)
        all_poly_placements.append(list(placements))

    # 5. Backtracking search for a valid tiling
    p1_placements, p2_placements, p3_placements = all_poly_placements

    for p1 in p1_placements:
        for p2 in p2_placements:
            if p1.isdisjoint(p2):
                combined_1_2 = p1.union(p2)
                for p3 in p3_placements:
                    if combined_1_2.isdisjoint(p3):
                        # Since total cell count is 16, a disjoint placement
                        # of all three guarantees a full tiling.
                        print("Yes")
                        return

    # If the loops complete without finding a solution
    print("No")

solve()