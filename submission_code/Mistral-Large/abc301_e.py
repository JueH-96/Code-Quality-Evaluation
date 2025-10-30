import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
T = int(data[2])

grid = [data[3 + i:3 + i + W] for i in range(0, H * W, W)]

# Find the start and goal positions
start = goal = None
candies = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)
        elif grid[i][j] == 'o':
            candies.append((i, j))

# BFS to find the maximum number of candies that can be collected
def bfs(start, goal, T):
    queue = deque([(start[0], start[1], 0, set())])
    visited = set()
    visited.add((start[0], start[1]))
    max_candies = -1

    while queue:
        x, y, moves, candy_set = queue.popleft()

        if (x, y) == goal:
            max_candies = max(max_candies, len(candy_set))

        if moves == T:
            continue

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in visited:
                visited.add((nx, ny))
                new_candy_set = set(candy_set)
                if grid[nx][ny] == 'o':
                    new_candy_set.add((nx, ny))
                queue.append((nx, ny, moves + 1, new_candy_set))

    return max_candies

result = bfs(start, goal, T)
print(result)