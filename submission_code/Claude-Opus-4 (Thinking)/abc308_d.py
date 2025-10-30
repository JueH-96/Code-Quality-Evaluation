from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

pattern = "snuke"

# BFS with state: (row, col, pattern_index)
queue = deque()

# Check if starting position has 's' (first character of pattern)
if grid[0][0] == 's':
    queue.append((0, 0, 0))

# Track visited states to avoid cycles
visited = [[[False for _ in range(5)] for _ in range(W)] for _ in range(H)]
if grid[0][0] == 's':
    visited[0][0][0] = True

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

found = False
while queue and not found:
    row, col, pattern_idx = queue.popleft()
    
    # Check if we reached the destination
    if row == H - 1 and col == W - 1:
        found = True
        break
    
    # Try all 4 directions
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        # Check bounds
        if 0 <= new_row < H and 0 <= new_col < W:
            # Next position in pattern (cycles back to 0 after 4)
            next_pattern_idx = (pattern_idx + 1) % 5
            
            # Check if the cell has the expected character
            if grid[new_row][new_col] == pattern[next_pattern_idx]:
                # Check if this state hasn't been visited
                if not visited[new_row][new_col][next_pattern_idx]:
                    visited[new_row][new_col][next_pattern_idx] = True
                    queue.append((new_row, new_col, next_pattern_idx))

print("Yes" if found else "No")