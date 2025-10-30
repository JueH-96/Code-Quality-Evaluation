# YOUR CODE HERE
def read_input():
    grids = []
    for _ in range(3):
        grid = []
        for _ in range(4):
            grid.append(input().strip())
        grids.append(grid)
    return grids

def extract_shape(grid):
    """Extract the coordinates of # cells in the polyomino"""
    shape = []
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '#':
                shape.append((i, j))
    return shape

def normalize_shape(shape):
    """Normalize shape to start from (0, 0)"""
    if not shape:
        return shape
    min_r = min(r for r, c in shape)
    min_c = min(c for r, c in shape)
    return [(r - min_r, c - min_c) for r, c in shape]

def rotate_90(shape):
    """Rotate shape 90 degrees clockwise"""
    # For rotation: (r, c) -> (c, -r)
    rotated = [(c, -r) for r, c in shape]
    return normalize_shape(rotated)

def get_all_rotations(shape):
    """Get all 4 rotations of a shape"""
    rotations = []
    current = shape
    for _ in range(4):
        # Add current rotation if it's not already in the list
        # We need to check if the set of coordinates is the same
        current_set = set(current)
        is_duplicate = False
        for rot in rotations:
            if set(rot) == current_set:
                is_duplicate = True
                break
        if not is_duplicate:
            rotations.append(current)
        current = rotate_90(current)
    
    return rotations

def can_place(shape, r_offset, c_offset, grid):
    """Check if shape can be placed at given offset on grid"""
    for r, c in shape:
        new_r = r + r_offset
        new_c = c + c_offset
        if new_r < 0 or new_r >= 4 or new_c < 0 or new_c >= 4:
            return False
        if grid[new_r][new_c] != 0:
            return False
    return True

def place_shape(shape, r_offset, c_offset, grid, value):
    """Place shape on grid with given value"""
    for r, c in shape:
        grid[r + r_offset][c + c_offset] = value

def remove_shape(shape, r_offset, c_offset, grid):
    """Remove shape from grid"""
    for r, c in shape:
        grid[r + r_offset][c + c_offset] = 0

def solve(polyominoes, grid, poly_idx):
    """Try to place all polyominoes using backtracking"""
    if poly_idx == 3:
        # Check if all cells are filled
        for i in range(4):
            for j in range(4):
                if grid[i][j] == 0:
                    return False
        return True
    
    shape = polyominoes[poly_idx]
    rotations = get_all_rotations(shape)
    
    for rotation in rotations:
        for r_offset in range(4):
            for c_offset in range(4):
                if can_place(rotation, r_offset, c_offset, grid):
                    place_shape(rotation, r_offset, c_offset, grid, poly_idx + 1)
                    if solve(polyominoes, grid, poly_idx + 1):
                        return True
                    remove_shape(rotation, r_offset, c_offset, grid)
    
    return False

def main():
    # Read input
    grids = read_input()
    
    # Extract shapes
    polyominoes = []
    for grid in grids:
        shape = extract_shape(grid)
        shape = normalize_shape(shape)
        polyominoes.append(shape)
    
    # Check if total number of cells equals 16
    total_cells = sum(len(poly) for poly in polyominoes)
    if total_cells != 16:
        print("No")
        return
    
    # Try to solve
    grid = [[0 for _ in range(4)] for _ in range(4)]
    if solve(polyominoes, grid, 0):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()