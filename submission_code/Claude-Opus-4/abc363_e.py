# YOUR CODE HERE
from collections import deque

H, W, Y = map(int, input().split())
grid = []
for _ in range(H):
    row = list(map(int, input().split()))
    grid.append(row)

# Track which sections have sunk
sunk = [[False] * W for _ in range(H)]

# Directions for adjacent cells (up, down, left, right)
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def is_valid(i, j):
    return 0 <= i < H and 0 <= j < W

def sink_sections(sea_level):
    # Queue for BFS
    queue = deque()
    
    # First, add all edge sections that should sink
    # Top and bottom edges
    for j in range(W):
        if not sunk[0][j] and grid[0][j] <= sea_level:
            queue.append((0, j))
            sunk[0][j] = True
        if not sunk[H-1][j] and grid[H-1][j] <= sea_level:
            queue.append((H-1, j))
            sunk[H-1][j] = True
    
    # Left and right edges
    for i in range(1, H-1):
        if not sunk[i][0] and grid[i][0] <= sea_level:
            queue.append((i, 0))
            sunk[i][0] = True
        if not sunk[i][W-1] and grid[i][W-1] <= sea_level:
            queue.append((i, W-1))
            sunk[i][W-1] = True
    
    # Also check sections adjacent to already sunk sections
    for i in range(H):
        for j in range(W):
            if sunk[i][j]:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if is_valid(ni, nj) and not sunk[ni][nj] and grid[ni][nj] <= sea_level:
                        queue.append((ni, nj))
                        sunk[ni][nj] = True
    
    # BFS to sink all connected sections
    while queue:
        i, j = queue.popleft()
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if is_valid(ni, nj) and not sunk[ni][nj] and grid[ni][nj] <= sea_level:
                sunk[ni][nj] = True
                queue.append((ni, nj))

# Process each year
for year in range(1, Y + 1):
    sea_level = year
    sink_sections(sea_level)
    
    # Count remaining sections
    remaining = 0
    for i in range(H):
        for j in range(W):
            if not sunk[i][j]:
                remaining += 1
    
    print(remaining)