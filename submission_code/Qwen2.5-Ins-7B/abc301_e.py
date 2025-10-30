# YOUR CODE HERE
from collections import deque

H, W, T = map(int, input().split())
grid = [input() for _ in range(H)]
start = None
goal = None
candies = []

for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)
        elif grid[i][j] == 'o':
            candies.append((i, j))

if start is None or goal is None:
    raise ValueError("Start or goal not found")

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def bfs(start, candies):
    queue = deque([(start, 0, set())])
    visited = set([start])
    max_candies = -1

    while queue:
        (x, y), moves, visited_candies = queue.popleft()
        if (x, y) == goal:
            max_candies = max(max_candies, len(visited_candies))
            continue

        if moves > T:
            continue

        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in visited:
                new_visited_candies = visited_candies.copy()
                if (nx, ny) in candies:
                    new_visited_candies.add((nx, ny))
                queue.append(((nx, ny), moves + 1, new_visited_candies))
                visited.add((nx, ny))

    return max_candies

max_candies = bfs(start, candies)
print(max_candies)