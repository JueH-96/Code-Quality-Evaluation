# YOUR CODE HERE
from collections import deque

def bfs(grid, start, goal, T):
    H, W = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], 0, set())])
    visited = set()
    max_candies = -1

    while queue:
        x, y, moves, candies = queue.popleft()

        if moves > T:
            continue

        if (x, y) == goal:
            max_candies = max(max_candies, len(candies))
            continue

        if (x, y, tuple(sorted(candies))) in visited:
            continue

        visited.add((x, y, tuple(sorted(candies))))

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                new_candies = set(candies)
                if grid[nx][ny] == 'o':
                    new_candies.add((nx, ny))
                queue.append((nx, ny, moves + 1, new_candies))

    return max_candies

H, W, T = map(int, input().split())
grid = [input().strip() for _ in range(H)]

start = goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)

result = bfs(grid, start, goal, T)
print(result)