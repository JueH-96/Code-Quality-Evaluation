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

# BFS
# State: (row, col, energy)
# visited[row][col] = max energy we've had at this position
visited = {}
queue = deque([(start[0], start[1], 0)])
visited[(start[0], start[1])] = 0

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
found = False

while queue and not found:
    row, col, energy = queue.popleft()
    
    # Check if we reached the goal
    if (row, col) == goal:
        found = True
        break
    
    # Skip if we've been here with more energy
    if (row, col) in visited and visited[(row, col)] > energy:
        continue
    
    # Try using medicine at current position
    if (row, col) in medicines:
        new_energy = medicines[(row, col)]
        if (row, col) not in visited or visited[(row, col)] < new_energy:
            visited[(row, col)] = new_energy
            queue.append((row, col, new_energy))
    
    # Try moving to adjacent cells
    if energy > 0:
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < H and 0 <= new_col < W:
                # Check if cell is not an obstacle
                if grid[new_row][new_col] != '#':
                    new_energy = energy - 1
                    
                    # Only proceed if we haven't been here with more energy
                    if (new_row, new_col) not in visited or visited[(new_row, new_col)] < new_energy:
                        visited[(new_row, new_col)] = new_energy
                        queue.append((new_row, new_col, new_energy))

print("Yes" if found else "No")