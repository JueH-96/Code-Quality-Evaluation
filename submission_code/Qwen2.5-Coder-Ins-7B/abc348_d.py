import sys
from collections import deque

# Read input
input = sys.stdin.read
data = input().split()

# Parse grid dimensions
H = int(data[0])
W = int(data[1])

# Parse grid
grid = []
index = 2
for i in range(H):
    row = list(data[index])
    grid.append(row)
    index += 1

# Parse medicines
N = int(data[index])
medicines = []
index += 1
for i in range(N):
    R = int(data[index]) - 1
    C = int(data[index + 1]) - 1
    E = int(data[index + 2])
    medicines.append((R, C, E))
    index += 3

# Find start and goal positions
start = None
goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            goal = (i, j)

# BFS to find the shortest path with medicine usage
def bfs():
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)
    
    while queue:
        (x, y), energy = queue.popleft()
        
        # Check if we have reached the goal
        if (x, y) == goal:
            return True
        
        # Check all four possible directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in visited:
                if energy > 0:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), energy - 1))
        
        # Check all medicines
        for R, C, E in medicines:
            if (x, y) == (R, C):
                visited.add((R, C))
                queue.append(((R, C), E))
    
    return False

# Determine if Takahashi can reach the goal
if bfs():
    print("Yes")
else:
    print("No")