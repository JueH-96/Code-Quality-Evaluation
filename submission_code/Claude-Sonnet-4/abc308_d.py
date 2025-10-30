from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

pattern = "snuke"

# BFS with state (row, col, pattern_index)
queue = deque()
visited = set()

# Start from (0,0) - need first character 's'
if grid[0][0] == 's':
    queue.append((0, 0, 0))  # pattern_index 0 means we just used 's'
    visited.add((0, 0, 0))

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

found = False
while queue and not found:
    row, col, pattern_idx = queue.popleft()
    
    # Check if we reached destination
    if row == H-1 and col == W-1:
        found = True
        break
    
    # Try all 4 directions
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        if 0 <= new_row < H and 0 <= new_col < W:
            # Next character we need
            next_pattern_idx = (pattern_idx + 1) % 5
            needed_char = pattern[next_pattern_idx]
            
            if grid[new_row][new_col] == needed_char:
                state = (new_row, new_col, next_pattern_idx)
                
                if state not in visited:
                    visited.add(state)
                    queue.append(state)

print("Yes" if found else "No")