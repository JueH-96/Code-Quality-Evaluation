from collections import deque

# Read input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]
N = int(input())
medicines = [tuple(map(int, input().split())) for _ in range(N)]

# Find start and goal positions
start = end = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            end = (i, j)

# Directions for movement
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS to check if we can reach the goal
def bfs():
    queue = deque([(start[0], start[1], 0, frozenset())])
    visited = set()
    visited.add((start[0], start[1], 0, frozenset()))

    while queue:
        x, y, energy, used_medicines = queue.popleft()

        # Check if we reached the goal
        if (x, y) == end:
            return True

        # Try using medicines
        for i in range(N):
            if (medicines[i][0] - 1, medicines[i][1] - 1) == (x, y) and i not in used_medicines:
                new_energy = medicines[i][2]
                new_used_medicines = used_medicines | {i}
                if (x, y, new_energy, new_used_medicines) not in visited:
                    visited.add((x, y, new_energy, new_used_medicines))
                    queue.append((x, y, new_energy, new_used_medicines))

        # Try moving in all directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy > 0:
                if (nx, ny, energy - 1, used_medicines) not in visited:
                    visited.add((nx, ny, energy - 1, used_medicines))
                    queue.append((nx, ny, energy - 1, used_medicines))

    return False

# Output the result
print("Yes" if bfs() else "No")