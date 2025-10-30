from collections import deque

H, W, Y = map(int, input().split())
grid = []
for i in range(H):
    row = list(map(int, input().split()))
    grid.append(row)

# Track which cells are sunk
sunk = [[False] * W for _ in range(H)]

# For each year
for year in range(1, Y + 1):
    sea_level = year
    
    # Find all cells that will sink this year
    queue = deque()
    new_sunk = [[False] * W for _ in range(H)]
    
    # Check all cells
    for i in range(H):
        for j in range(W):
            if not sunk[i][j] and grid[i][j] <= sea_level:
                # Check if this cell is adjacent to the sea
                adjacent_to_sea = False
                
                # Check if on border
                if i == 0 or i == H - 1 or j == 0 or j == W - 1:
                    adjacent_to_sea = True
                else:
                    # Check if adjacent to a sunk cell
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and sunk[ni][nj]:
                            adjacent_to_sea = True
                            break
                
                if adjacent_to_sea:
                    new_sunk[i][j] = True
                    queue.append((i, j))
    
    # BFS to propagate sinking
    while queue:
        i, j = queue.popleft()
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not sunk[ni][nj] and not new_sunk[ni][nj] and grid[ni][nj] <= sea_level:
                new_sunk[ni][nj] = True
                queue.append((ni, nj))
    
    # Update sunk status
    for i in range(H):
        for j in range(W):
            if new_sunk[i][j]:
                sunk[i][j] = True
    
    # Count remaining cells
    remaining = sum(1 for i in range(H) for j in range(W) if not sunk[i][j])
    print(remaining)