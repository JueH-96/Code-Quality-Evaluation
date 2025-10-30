# YOUR CODE HERE
from collections import deque

def bfs(grid, i, j, H, W):
    queue = deque([(i, j)])
    visited = [[False] * W for _ in range(H)]
    visited[i][j] = True
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] != '#':
                visited[nx][ny] = True
                queue.append((nx, ny))
    return count

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]
max_degree_of_freedom = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] != '#':
            max_degree_of_freedom = max(max_degree_of_freedom, bfs(grid, i, j, H, W))
print(max_degree_of_freedom)