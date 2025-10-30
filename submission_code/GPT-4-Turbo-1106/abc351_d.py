from collections import deque

def bfs(grid, start, visited):
    queue = deque([start])
    visited[start[0]][start[1]] = True
    freedom_degree = 1
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not visited[nx][ny] and grid[nx][ny] == '.':
                freedom_degree += 1
                visited[nx][ny] = True
                queue.append((nx, ny))
    
    return freedom_degree

def max_degree_of_freedom(grid):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    max_freedom = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not visited[i][j]:
                max_freedom = max(max_freedom, bfs(grid, (i, j), visited))
    
    return max_freedom

# Read input
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Calculate and output the answer
print(max_degree_of_freedom(grid))