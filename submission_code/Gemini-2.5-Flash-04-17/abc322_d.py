import sys
from itertools import permutations

def parse_polyomino(lines):
    """
    Parses a 4x4 input grid for a polyomino shape.
    Returns a list of (row, col) tuples representing the shape, normalized and sorted.
    """
    cells = []
    for r in range(4):
        for c in range(4):
            if lines[r][c] == '#':
                cells.append((r, c))

    if not cells:
        # As per constraints, polyominoes are not empty.
        return []

    # Normalize the shape by shifting min_r and min_c to (0,0)
    min_r = min(cell[0] for cell in cells)
    min_c = min(cell[1] for cell in cells)
    normalized_cells = [(cell[0] - min_r, cell[1] - min_c) for cell in cells]

    # Sort the cells to get a unique representation for a given shape
    normalized_cells.sort()
    return normalized_cells

def rotate_shape(shape):
    """
    Rotates a normalized shape (list of (r, c) tuples) 90 degrees clockwise.
    Returns the new shape, normalized and sorted.
    """
    if not shape:
        return []

    # Find dimensions (H, W) of the current normalized shape's bounding box.
    # Since shape is normalized, min row is 0, min col is 0.
    max_r_prime = max(cell[0] for cell in shape)
    # max_c_prime = max(cell[1] for cell in shape) # Not directly used in rotation formula derivation
    H = max_r_prime + 1

    # Rotate each point (r', c') by 90 degrees clockwise around (0,0): (c', -r').
    rotated_cells_raw = [(cell[1], -cell[0]) for cell in shape]

    # Normalize the rotated shape.
    # The rotated shape's minimum row is min(c') = 0.
    # The rotated shape's minimum col is min(-r') = -max(r') = -(H-1).
    min_r_rotated = min(cell[0] for cell in rotated_cells_raw) # This is actually min c'
    min_c_rotated = min(cell[1] for cell in rotated_cells_raw) # This is actually min -r'

    # Shift the rotated points by (-min_r_rotated, -min_c_rotated)
    normalized_rotated_cells = [(cell[0] - min_r_rotated, cell[1] - min_c_rotated) for cell in rotated_cells_raw]
    # This is equivalent to [(c', -r' - (-(H-1))) for (c', -r') in rotated_cells_raw]
    # which simplifies to [(c', -r' + H - 1)].

    normalized_rotated_cells.sort()
    return normalized_rotated_cells

def get_all_shapes(initial_shape):
    """
    Generates all distinct normalized rotations (0, 90, 180, 270 degrees)
    of the given initial normalized shape.
    Returns a list of distinct normalized shapes (each a sorted list of tuples).
    """
    shapes = [initial_shape]
    current_shape = initial_shape
    # Rotate up to 3 times. The 4th rotation brings it back to the original.
    for _ in range(3):
        rotated = rotate_shape(current_shape)
        # Add the rotated shape if it's distinct from shapes already found
        if rotated not in shapes:
             shapes.append(rotated)
        current_shape = rotated # Continue rotating from the previous result
    return shapes

def solve():
    """
    Reads the three polyomino inputs, finds all their shapes, and uses
    backtracking to check if they can fill a 4x4 grid without overlap.
    """
    polyominoes_input = []
    for _ in range(3):
        lines = [sys.stdin.readline().strip() for _ in range(4)]
        polyominoes_input.append(lines)

    all_polyomino_shapes = []
    for p_input in polyominoes_input:
        initial_shape = parse_polyomino(p_input)
        # Constraints guarantee non-empty polyominoes
        if not initial_shape:
             # This case should not be reachable based on constraints,
             # but handle defensively if an empty polyomino was somehow input.
             print("No") # Cannot form a 16-square grid with an empty piece
             return
        all_polyomino_shapes.append(get_all_shapes(initial_shape))

    # Sanity check: Total number of cells across all pieces must be 16
    total_area = sum(len(shapes[0]) for shapes in all_polyomino_shapes)
    if total_area != 16:
        print("No")
        return

    def can_place(shape, dr, dc, occupied_cells):
        """
        Checks if a shape can be placed at (dr, dc) offset on the grid
        without overlapping with existing occupied_cells.
        Assumes the shape placed at (dr, dc) fits within the 4x4 bounds
        (this is ensured by the calling loop for dr, dc).
        """
        for r_prime, c_prime in shape:
            r = r_prime + dr
            c = c_prime + dc
            # Check for overlap with already occupied cells
            if (r, c) in occupied_cells:
                return False # Overlap detected
        return True # No overlap

    def place_and_solve(poly_indices_order, occupied_cells):
        """
        Recursive backtracking function.
        Attempts to place the polyominoes specified by `poly_indices_order`
        onto the grid, starting from the first index, given the `occupied_cells`.
        """
        # Base case: If no polyominoes are left to place
        if not poly_indices_order:
            # Check if the grid is completely filled (16 squares)
            return len(occupied_cells) == 16

        # Take the next polyomino index from the order
        current_poly_idx = poly_indices_order[0]
        remaining_poly_indices = poly_indices_order[1:]

        # Get all possible shapes for the current polyomino
        shapes_of_current_poly = all_polyomino_shapes[current_poly_idx]

        # Try placing the polyomino in each of its possible shapes
        for shape in shapes_of_current_poly:
            # Determine the dimensions of the normalized shape's bounding box
            max_r_prime = max(cell[0] for cell in shape)
            max_c_prime = max(cell[1] for cell in shape)

            # Iterate through all possible top-left placement positions (dr, dc)
            # such that the shape remains entirely within the 4x4 grid.
            # The shape spans rows 0 to max_r_prime and columns 0 to max_c_prime
            # in its normalized form.
            # If placed at (dr, dc), it spans rows dr to dr + max_r_prime
            # and columns dc to dc + max_c_prime.
            # We need dr + max_r_prime < 4 and dc + max_c_prime < 4.
            # So, dr <= 3 - max_r_prime and dc <= 3 - max_c_prime.
            # Possible values for dr are 0, 1, ..., 3 - max_r_prime.
            # Possible values for dc are 0, 1, ..., 3 - max_c_prime.
            # The ranges `range(4 - max_r_prime)` and `range(4 - max_c_prime)` are always valid
            # because max_r_prime and max_c_prime are at most 3 (for a 4x4 grid)
            # and at least 0 (for a non-empty normalized shape).

            for dr in range(4 - max_r_prime):
                for dc in range(4 - max_c_prime):
                    # Check if this placement is valid (no overlap)
                    if can_place(shape, dr, dc, occupied_cells):
                        # If valid, create the set of cells covered by this placement
                        placement_cells = set((r_prime + dr, c_prime + dc) for r_prime, c_prime in shape)

                        # Recursively call with the next polyomino and the updated occupied cells
                        if place_and_solve(remaining_poly_indices, occupied_cells.union(placement_cells)):
                            return True # If the recursive call found a solution, we are done

        # If no shape or no placement for the current polyomino leads to a solution
        return False

    # Start the search by trying all possible orderings of the three polyominoes.
    # The initial state is an empty grid (empty set of occupied cells).
    for perm in permutations([0, 1, 2]):
        if place_and_solve(list(perm), set()):
            # Found a solution for this permutation
            print("Yes")
            return

    # If no permutation led to a solution after trying all orders
    print("No")

# Execute the solve function
solve()