def parse_polyomino(lines):
    """Extract occupied positions from a 4x4 grid representation"""
    positions = []
    for i in range(4):
        for j in range(4):
            if lines[i][j] == '#':
                positions.append((i, j))
    return positions

def normalize_polyomino(positions):
    """Normalize polyomino by moving it to start from (0,0)"""
    if not positions:
        return positions
    min_r = min(pos[0] for pos in positions)
    min_c = min(pos[1] for pos in positions)
    return [(r - min_r, c - min_c) for r, c in positions]

def rotate_90(positions):
    """Rotate polyomino 90 degrees clockwise"""
    return [(c, -r) for r, c in positions]

def get_all_rotations(positions):
    """Get all 4 rotations of a polyomino"""
    rotations = []
    current = positions
    for _ in range(4):
        normalized = normalize_polyomino(current)
        if normalized not in rotations:
            rotations.append(normalized)
        current = rotate_90(current)
    return rotations

def can_place(polyomino, start_r, start_c, grid):
    """Check if polyomino can be placed at given position"""
    for r, c in polyomino:
        new_r, new_c = start_r + r, start_c + c
        if new_r < 0 or new_r >= 4 or new_c < 0 or new_c >= 4:
            return False
        if grid[new_r][new_c]:
            return False
    return True

def place_polyomino(polyomino, start_r, start_c, grid, value):
    """Place or remove polyomino on grid"""
    for r, c in polyomino:
        new_r, new_c = start_r + r, start_c + c
        grid[new_r][new_c] = value

def is_grid_full(grid):
    """Check if all positions in grid are occupied"""
    for row in grid:
        for cell in row:
            if not cell:
                return False
    return True

def solve(polyominoes):
    """Try to fill 4x4 grid with all polyominoes"""
    # Generate all rotations for each polyomino
    all_rotations = []
    for poly in polyominoes:
        all_rotations.append(get_all_rotations(poly))
    
    grid = [[False] * 4 for _ in range(4)]
    
    def backtrack(poly_idx):
        if poly_idx == 3:
            return is_grid_full(grid)
        
        for rotation in all_rotations[poly_idx]:
            for start_r in range(4):
                for start_c in range(4):
                    if can_place(rotation, start_r, start_c, grid):
                        place_polyomino(rotation, start_r, start_c, grid, True)
                        if backtrack(poly_idx + 1):
                            return True
                        place_polyomino(rotation, start_r, start_c, grid, False)
        return False
    
    return backtrack(0)

# Read input
lines = []
for _ in range(12):
    lines.append(input().strip())

# Parse the three polyominoes
polyominoes = []
for i in range(3):
    poly_lines = lines[i*4:(i+1)*4]
    positions = parse_polyomino(poly_lines)
    normalized = normalize_polyomino(positions)
    polyominoes.append(normalized)

# Solve and output result
if solve(polyominoes):
    print("Yes")
else:
    print("No")