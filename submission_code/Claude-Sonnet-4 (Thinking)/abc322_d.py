def rotate_90(grid):
    """Rotate 4x4 grid 90 degrees clockwise"""
    n = 4
    rotated = [['.' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated[j][n-1-i] = grid[i][j]
    return rotated

def get_coords(grid):
    """Get coordinates of '#' squares"""
    return [(i, j) for i in range(4) for j in range(4) if grid[i][j] == '#']

def normalize(coords):
    """Normalize coordinates to start from (0,0)"""
    if not coords:
        return []
    min_r = min(r for r, c in coords)
    min_c = min(c for r, c in coords)
    return sorted([(r - min_r, c - min_c) for r, c in coords])

def get_orientations(polyomino):
    """Get all unique orientations"""
    orientations = []
    seen = set()
    current = polyomino
    
    for _ in range(4):
        coords = get_coords(current)
        normalized = tuple(normalize(coords))
        if normalized not in seen:
            seen.add(normalized)
            orientations.append(list(normalized))
        current = rotate_90(current)
    
    return orientations

def can_place(board, shape, r, c):
    """Check if shape can be placed at (r,c)"""
    for dr, dc in shape:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 4 and 0 <= nc < 4) or board[nr][nc]:
            return False
    return True

def place(board, shape, r, c, piece_id):
    """Place shape on board"""
    new_board = [row[:] for row in board]
    for dr, dc in shape:
        new_board[r + dr][c + dc] = piece_id
    return new_board

def solve(polyominoes):
    orientations = [get_orientations(poly) for poly in polyominoes]
    
    def backtrack(board, piece):
        if piece == 3:
            return all(board[i][j] for i in range(4) for j in range(4))
        
        for orientation in orientations[piece]:
            for r in range(4):
                for c in range(4):
                    if can_place(board, orientation, r, c):
                        new_board = place(board, orientation, r, c, piece + 1)
                        if backtrack(new_board, piece + 1):
                            return True
        return False
    
    return backtrack([[0] * 4 for _ in range(4)], 0)

# Read input
polyominoes = []
for _ in range(3):
    poly = []
    for _ in range(4):
        poly.append(list(input().strip()))
    polyominoes.append(poly)

# Solve and output
print("Yes" if solve(polyominoes) else "No")