from collections import deque, defaultdict
import sys

def bfs(grid, start, goal, H, W):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == goal:
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in visited:
                visited.add((nx, ny))
                distances[(nx, ny)] = dist + 1
                queue.append(((nx, ny), dist + 1))
    return float('inf')

def solve(H, W, T, grid):
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

    if bfs(grid, start, goal, H, W) > T:
        return -1

    def dfs(path, visited, current_dist):
        if current_dist > T:
            return 0
        if path[-1] == goal:
            return len(visited)
        max_candies = 0
        for candy in candies:
            if candy not in visited:
                new_path = path + [candy]
                new_visited = visited | {candy}
                new_dist = current_dist + bfs(grid, path[-1], candy, H, W)
                max_candies = max(max_candies, dfs(new_path, new_visited, new_dist))
        return max_candies

    return dfs([start], set(), 0)

# Read input
H, W, T = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Solve and print the result
print(solve(H, W, T, grid))