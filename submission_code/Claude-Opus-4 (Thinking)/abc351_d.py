from collections import deque

def has_adjacent_magnet(grid, i, j, H, W):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
            return True
    return False

def bfs(grid, start_i, start_j, H, W, computed):
    if (start_i, start_j) in computed:
        return computed[(start_i, start_j)]
    
    visited = set()
    queue = deque([(start_i, start_j)])
    visited.add((start_i, start_j))
    
    while queue:
        i, j = queue.popleft()
        
        # Check if current cell has adjacent magnet
        if has_adjacent_magnet(grid, i, j, H, W):
            # Cannot move from this cell
            continue
        
        # Can move to adjacent cells
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in visited and grid[ni][nj] == '.':
                visited.add((ni, nj))
                queue.append((ni, nj))
    
    # All cells in visited have the same degree of freedom
    degree = len(visited)
    for cell in visited:
        computed[cell] = degree
    
    return degree

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

max_freedom = 0
computed = {}
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            freedom = bfs(grid, i, j, H, W, computed)
            max_freedom = max(max_freedom, freedom)

print(max_freedom)