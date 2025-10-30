import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])

grid = []
index = 2
for _ in range(H):
    grid.append(data[index])
    index += 1

N = int(data[index])
index += 1

medicines = {}
for _ in range(N):
    R = int(data[index]) - 1
    C = int(data[index + 1]) - 1
    E = int(data[index + 2])
    index += 3
    medicines[(R, C)] = E

# Find start (S) and target (T) positions
start = None
target = None
for r in range(H):
    for c in range(W):
        if grid[r][c] == 'S':
            start = (r, c)
        elif grid[r][c] == 'T':
            target = (r, c)

# Directions for movement (right, left, down, up)
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# BFS with state (row, col, energy)
queue = deque([(start[0], start[1], 0)])
visited = set()
visited.add((start[0], start[1], 0))

while queue:
    r, c, energy = queue.popleft()
    
    # Check if we reached the target
    if (r, c) == target:
        print("Yes")
        sys.exit(0)
    
    # Use medicine if available at this cell
    if (r, c) in medicines:
        new_energy = medicines[(r, c)]
        if new_energy > energy:
            energy = new_energy
            if (r, c, energy) not in visited:
                visited.add((r, c, energy))
                queue.append((r, c, energy))
    
    # Try to move to adjacent cells
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
            if energy > 0:
                if (nr, nc, energy - 1) not in visited:
                    visited.add((nr, nc, energy - 1))
                    queue.append((nr, nc, energy - 1))

print("No")