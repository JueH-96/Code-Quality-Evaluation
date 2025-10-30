import sys

GRID_SIZE = 4
TOTAL_CELLS = GRID_SIZE * GRID_SIZE

def normalize(points):
    """
    Normalizes a set of points (relative coordinates of a polyomino)
    by shifting them so that the top-leftmost occupied cell is at (0,0).
    Returns a frozenset for hashability and uniqueness.
    """
    if not points:
        return frozenset()
    min_r = min(p[0] for p in points)
    min_c = min(p[1] for p in points)
    return frozenset((r - min_r, c - min_c) for r, c in points)

def rotate_point(r, c, N=GRID_SIZE):
    """
    Rotates a point (r, c) 90 degrees clockwise within an N x N grid.
    The new coordinates (r', c') are (c, N - 1 - r).
    """
    return (c, N - 1 - r)

def generate_all_placements(original_poly_points):
    """
    Generates all unique valid absolute placements (positions on the 4x4 grid)
    for a given polyomino, considering all its rotations.
    """
    all_rotations_normalized = set()
    current_shape_points_for_rotation = list(original_poly_points) # Start with original points as a list

    # Generate unique normalized shapes for all 4 rotations (0, 90, 180, 270 degrees)
    for _ in range(4):
        normalized_shape = normalize(current_shape_points_for_rotation)
        all_rotations_normalized.add(normalized_shape)
        
        # Prepare for the next rotation: apply rotation to current points
        current_shape_points_for_rotation = [rotate_point(r, c) for r, c in current_shape_points_for_rotation]
        
    placements = []
    for rotated_shape_norm in all_rotations_normalized:
        # According to problem constraints, polyominoes are not empty.
        # This check is mostly for robustness.
        if not rotated_shape_norm:
            continue 

        # Calculate the current extent of the normalized shape.
        # This is used to determine the maximum possible top-left offset (r_offset, c_offset)
        # such that the entire translated shape fits within the GRID_SIZE x GRID_SIZE area.
        max_dr = 0
        max_dc = 0
        # These max() calls would fail for an empty set, but we already checked for that.
        max_dr = max(p[0] for p in rotated_shape_norm)
        max_dc = max(p[1] for p in rotated_shape_norm)

        # Iterate over all valid top-left offsets for this normalized shape
        for r_offset in range(GRID_SIZE - max_dr):
            for c_offset in range(GRID_SIZE - max_dc):
                # Translate the normalized shape to its absolute position on the grid
                translated_shape = frozenset((r_offset + r, c_offset + c) for r, c in rotated_shape_norm)
                placements.append(translated_shape)
    
    return placements


# --- Main execution ---

# 1. Read input and store polyomino raw points
polyomino_inputs_raw = []
for _ in range(3):
    current_poly_points = set()
    for r in range(GRID_SIZE):
        line = sys.stdin.readline().strip()
        for c in range(GRID_SIZE):
            if line[c] == '#':
                current_poly_points.add((r, c))
    polyomino_inputs_raw.append(frozenset(current_poly_points))

# 2. Pre-check: Sum of areas must be exactly TOTAL_CELLS (16)
# If this condition is not met, it's impossible to tile the grid perfectly.
total_cells_count = sum(len(p) for p in polyomino_inputs_raw)
if total_cells_count != TOTAL_CELLS:
    print("No")
    sys.exit()

# 3. Generate all possible placements for each polyomino.
# all_poly_placements[i] will be a list of frozensets, where each frozenset
# represents a distinct absolute placement of polyomino 'i' on the grid.
all_poly_placements = [generate_all_placements(p) for p in polyomino_inputs_raw]

# 4. Backtracking search to find a valid combination of placements
def solve(k, current_grid_occupied_set):
    """
    Recursively tries to place polyominoes.
    k: The index of the current polyomino to place (0, 1, or 2).
    current_grid_occupied_set: A frozenset of (r, c) tuples representing cells
                               already occupied by polyominoes 0 to k-1.
    """
    # Base case: All 3 polyominoes have been successfully placed
    if k == 3:
        # Check if all cells are covered (no overlaps are ensured by checks below)
        return len(current_grid_occupied_set) == TOTAL_CELLS

    # Recursive step: Try to place polyomino 'k'
    for placement in all_poly_placements[k]:
        # Check for overlap with already placed polyominoes
        is_overlap = False
        for r, c in placement:
            if (r, c) in current_grid_occupied_set:
                is_overlap = True
                break
        
        if is_overlap:
            continue # This placement overlaps, try the next one

        # If no overlap, try placing it and recurse for the next polyomino
        # Create a new set by uniting current occupied cells and the new placement
        new_grid_occupied_set = current_grid_occupied_set.union(placement)
        
        # Recurse: if a solution is found with this placement, return True
        if solve(k + 1, new_grid_occupied_set):
            return True 

    return False # No placement for polyomino 'k' leads to a solution


# Initial call to the backtracking function: start with polyomino 0 and an empty grid
if solve(0, frozenset()): 
    print("Yes")
else:
    print("No")