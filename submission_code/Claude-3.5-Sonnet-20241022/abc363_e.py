from collections import deque

def get_neighbors(i, j, H, W):
    neighbors = []
    for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
        if 0 <= ni < H and 0 <= nj < W:
            neighbors.append((ni, nj))
    return neighbors

def solve():
    H, W, Y = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(map(int, input().split())))
    
    # For each year
    for year in range(1, Y+1):
        # Track which cells are flooded
        flooded = [[False]*W for _ in range(H)]
        
        # Start with cells adjacent to border
        queue = deque()
        
        # Check border cells
        for i in range(H):
            for j in range(W):
                if i == 0 or i == H-1 or j == 0 or j == W-1:
                    if grid[i][j] <= year:
                        flooded[i][j] = True
                        queue.append((i,j))
        
        # BFS to flood connected cells
        while queue:
            i, j = queue.popleft()
            for ni, nj in get_neighbors(i, j, H, W):
                if not flooded[ni][nj] and grid[ni][nj] <= year:
                    flooded[ni][nj] = True
                    queue.append((ni, nj))
        
        # Count remaining cells
        remaining = sum(1 for i in range(H) for j in range(W) if not flooded[i][j])
        print(remaining)

solve()