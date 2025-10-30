# YOUR CODE HERE
from collections import deque

def can_reach_goal():
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    N = int(input())
    medicines = {}
    for _ in range(N):
        r, c, e = map(int, input().split())
        medicines[(r-1, c-1)] = e

    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)

    queue = deque([(start[0], start[1], 0)])
    visited = set()

    while queue:
        x, y, energy = queue.popleft()

        if (x, y) == goal:
            return "Yes"

        if (x, y, energy) in visited:
            continue

        visited.add((x, y, energy))

        if (x, y) in medicines:
            new_energy = medicines[(x, y)]
            queue.append((x, y, new_energy))

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy > 0:
                queue.append((nx, ny, energy - 1))

    return "No"

print(can_reach_goal())