from collections import deque

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

N = int(input())
medicines = {}
for _ in range(N):
    r, c, e = map(int, input().split())
    medicines[(r-1, c-1)] = e  # Convert to 0-indexed

# Find start and goal positions
start = None
goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            goal = (i, j)

# BFS with state (row, col, energy)
# We need to track visited states to avoid cycles
visited = set()
queue = deque()

# Check if there's a medicine at start position
if start in medicines:
    queue.append((start[0], start[1], medicines[start]))
    visited.add((start[0], start[1], medicines[start]))
else:
    queue.append((start[0], start[1], 0))
    visited.add((start[0], start[1], 0))

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

found = False
while queue and not found:
    r, c, energy = queue.popleft()
    
    # Check if we reached the goal
    if (r, c) == goal:
        found = True
        break
    
    # Try to move in all directions
    if energy > 0:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds and obstacles
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                new_energy = energy - 1
                
                # Check if there's a medicine at the new position
                if (nr, nc) in medicines:
                    # We can choose to use the medicine
                    med_energy = medicines[(nr, nc)]
                    if (nr, nc, med_energy) not in visited:
                        visited.add((nr, nc, med_energy))
                        queue.append((nr, nc, med_energy))
                
                # Also consider not using the medicine (or if there's no medicine)
                if (nr, nc, new_energy) not in visited:
                    visited.add((nr, nc, new_energy))
                    queue.append((nr, nc, new_energy))

if found:
    print("Yes")
else:
    print("No")