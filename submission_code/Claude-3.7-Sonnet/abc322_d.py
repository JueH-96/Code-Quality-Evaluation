def extract_polyomino(grid):
    """Extract the shape of the polyomino from a 4x4 grid."""
    coordinates = []
    for r in range(4):
        for c in range(4):
            if grid[r][c] == '#':
                coordinates.append((r, c))
    
    # Normalize the polyomino
    min_r = min(r for r, c in coordinates)
    min_c = min(c for r, c in coordinates)
    return [(r - min_r, c - min_c) for r, c in coordinates]

def rotate_90_degrees(shape):
    """Rotate a polyomino shape 90 degrees clockwise."""
    max_r = max(r for r, c in shape)
    # Rotate 90 degrees clockwise: (r, c) -> (c, max_r - r)
    rotated = [(c, max_r - r) for r, c in shape]
    
    # Normalize the rotated shape
    min_r = min(r for r, c in rotated)
    min_c = min(c for r, c in rotated)
    return [(r - min_r, c - min_c) for r, c in rotated]

def get_all_rotations(polyomino):
    """Generate all unique rotations of a polyomino."""
    rotations = [polyomino]
    
    current_rotation = polyomino
    for _ in range(3):  # At most 3 more rotations
        current_rotation = rotate_90_degrees(current_rotation)
        
        if not any(set(current_rotation) == set(r) for r in rotations):
            rotations.append(current_rotation)
    
    return rotations

def get_all_valid_placements(rotation, grid_size=4):
    """Generate all valid placements of a rotation on the grid."""
    placements = []
    
    # Find the dimensions of the rotation
    max_r = max(r for r, c in rotation)
    max_c = max(c for r, c in rotation)
    
    for r in range(grid_size - max_r):
        for c in range(grid_size - max_c):
            placement = {(r + dr, c + dc) for dr, dc in rotation}
            placements.append(placement)
    
    return placements

def is_valid_config(all_placements, index=0, placed_positions=None):
    """Check if there's a valid configuration using backtracking."""
    if placed_positions is None:
        placed_positions = set()
    
    if index == len(all_placements):
        # Check if all cells in the 4x4 grid are covered
        all_cells = {(r, c) for r in range(4) for c in range(4)}
        return placed_positions == all_cells
    
    for placement in all_placements[index]:
        # Check if the placement is valid
        if not placement.intersection(placed_positions):
            # Place the polyomino
            new_placed_positions = placed_positions.union(placement)
            
            # Recursive call
            if is_valid_config(all_placements, index + 1, new_placed_positions):
                return True
    
    return False

def solve():
    polyominoes = []
    
    for _ in range(3):
        grid = []
        for _ in range(4):
            row = input()
            grid.append(row)
        polyominoes.append(extract_polyomino(grid))
    
    all_placements = []
    for polyomino in polyominoes:
        rotations = get_all_rotations(polyomino)
        placements_for_polyomino = []
        
        for rotation in rotations:
            placements_for_polyomino.extend(get_all_valid_placements(rotation))
        
        all_placements.append(placements_for_polyomino)
    
    if is_valid_config(all_placements):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()