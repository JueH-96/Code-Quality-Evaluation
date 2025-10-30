def count_sensors(grid, H, W):
    visited = [[False] * W for _ in range(H)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == '#':
                    visited[nx][ny] = True
                    stack.append((nx, ny))

    sensor_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                visited[i][j] = True
                dfs(i, j)
                sensor_count += 1

    return sensor_count

import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
grid = data[1:H + 1]

result = count_sensors(grid, H, W)
print(result)