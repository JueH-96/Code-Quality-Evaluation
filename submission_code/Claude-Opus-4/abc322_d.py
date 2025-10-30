def read_polyomino():
    grid = []
    for _ in range(4):
        grid.append(input().strip())
    return grid

def extract_shape(grid):
    # Find all '#' positions
    positions = []
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '#':
                positions.append((i, j))
    
    if not positions:
        return []
    
    # Normalize to start from (0, 0)
    min_r = min(p[0] for p in positions)
    min_c = min(p[1] for p in positions)
    
    normalized = [(r - min_r, c - min_c) for r, c in positions]
    return sorted(normalized)

def rotate_90(shape):
    # Rotate 90 degrees clockwise
    if not shape:
        return []
    
    rotated = []
    for r, c in shape:
        rotated.append((c, -r))
    
    # Normalize
    min_r = min(p[0] for p in rotated)
    min_c = min(p[1] for p in rotated)
    
    normalized = [(r - min_r, c - min_c) for r, c in rotated]
    return sorted(normalized)

def get_all_rotations(shape):
    rotations = [shape]
    current = shape
    
    for _ in range(3):
        current = rotate_90(current)
        if current not in rotations:
            rotations.append(current)
    
    return rotations

def get_all_placements(shape):
    placements = []
    
    # Get all rotations
    rotations = get_all_rotations(shape)
    
    for rotation in rotations:
        # Get bounds
        if not rotation:
            continue
            
        max_r = max(p[0] for p in rotation)
        max_c = max(p[1] for p in rotation)
        
        # Try all valid translations
        for dr in range(4 - max_r):
            for dc in range(4 - max_c):
                placement = [(r + dr, c + dc) for r, c in rotation]
                placements.append(sorted(placement))
    
    return placements

def can_place(grid, polyomino):
    for r, c in polyomino:
        if grid[r][c]:
            return False
    return True

def place(grid, polyomino, value):
    for r, c in polyomino:
        grid[r][c] = value

def solve(shapes):
    # Get all possible placements for each polyomino
    all_placements = []
    for shape in shapes:
        all_placements.append(get_all_placements(shape))
    
    # Try all combinations
    grid = [[0] * 4 for _ in range(4)]
    
    for p1 in all_placements[0]:
        # Place first polyomino
        place(grid, p1, 1)
        
        for p2 in all_placements[1]:
            if can_place(grid, p2):
                # Place second polyomino
                place(grid, p2, 2)
                
                for p3 in all_placements[2]:
                    if can_place(grid, p3):
                        # Place third polyomino
                        place(grid, p3, 3)
                        
                        # Check if all cells are covered
                        all_covered = all(grid[i][j] != 0 for i in range(4) for j in range(4))
                        
                        if all_covered:
                            return True
                        
                        # Remove third polyomino
                        place(grid, p3, 0)
                
                # Remove second polyomino
                place(grid, p2, 0)
        
        # Remove first polyomino
        place(grid, p1, 0)
    
    return False

# Read input
polyominoes = []
for _ in range(3):
    polyominoes.append(read_polyomino())

# Extract shapes
shapes = []
for poly in polyominoes:
    shape = extract_shape(poly)
    shapes.append(shape)

# Check if total cells equals 16
total_cells = sum(len(shape) for shape in shapes)
if total_cells != 16:
    print("No")
else:
    # Solve
    if solve(shapes):
        print("Yes")
    else:
        print("No")