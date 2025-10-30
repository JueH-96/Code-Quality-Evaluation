import sys
from collections import deque

input = sys.stdin.read

def can_reach_goal(H, W, grid, N, medicines):
    start, goal = None, None

    # Find start and goal positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)

    # Directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS setup
    queue = deque([(start[0], start[1], 0)])  # (row, col, energy)
    visited = set((start[0], start[1], 0))
    medicine_used = set()

    while queue:
        x, y, energy = queue.popleft()

        # Check if we reached the goal
        if (x, y) == goal:
            return "Yes"

        # Try to use medicine if available
        if (x, y) in medicines and (x, y) not in medicine_used:
            new_energy = medicines[(x, y)]
            medicine_used.add((x, y))
            if (x, y, new_energy) not in visited:
                visited.add((x, y, new_energy))
                queue.append((x, y, new_energy))

        # Try moving in all directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy > 0:
                if (nx, ny, energy - 1) not in visited:
                    visited.add((nx, ny, energy - 1))
                    queue.append((nx, ny, energy - 1))

    return "No"

# Read input
data = input().split()
H = int(data[0])
W = int(data[1])
grid = [data[i] for i in range(2, 2 + H)]
N = int(data[2 + H])
medicines = {}
for i in range(N):
    R, C, E = int(data[3 + H + 3 * i]) - 1, int(data[4 + H + 3 * i]) - 1, int(data[5 + H + 3 * i])
    medicines[(R, C)] = E

# Solve and print the result
print(can_reach_goal(H, W, grid, N, medicines))