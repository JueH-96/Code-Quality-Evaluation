from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find start and goal
start = None
goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            goal = (i, j)

N = int(input())
medicines = {}
for _ in range(N):
    r, c, e = map(int, input().split())
    medicines[(r-1, c-1)] = e  # Convert to 0-indexed

# BFS with state (row, col, energy, used_medicines)
visited = set()
queue = deque()

# Start state
initial_state = (start[0], start[1], 0, frozenset())
queue.append(initial_state)
visited.add(initial_state)

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

while queue:
    row, col, energy, used_meds = queue.popleft()
    
    # Check if we reached the goal
    if (row, col) == goal:
        print("Yes")
        exit()
    
    # Try using medicine at current position
    if (row, col) in medicines and (row, col) not in used_meds:
        new_energy = medicines[(row, col)]
        new_used_meds = used_meds | {(row, col)}
        new_state = (row, col, new_energy, new_used_meds)
        if new_state not in visited:
            visited.add(new_state)
            queue.append(new_state)
    
    # Try moving to adjacent cells
    if energy > 0:
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] != '#':
                new_state = (new_row, new_col, energy - 1, used_meds)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)

print("No")