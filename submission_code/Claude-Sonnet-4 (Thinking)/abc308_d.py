from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

pattern = "snuke"
queue = deque()
visited = set()

# Start from (0,0) if it has 's'
if grid[0][0] == 's':
    queue.append((0, 0, 0))
    visited.add((0, 0, 0))

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
found = False

while queue and not found:
    row, col, pattern_idx = queue.popleft()
    
    if row == H - 1 and col == W - 1:
        found = True
        break
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        if 0 <= new_row < H and 0 <= new_col < W:
            next_pattern_idx = (pattern_idx + 1) % 5
            expected_char = pattern[next_pattern_idx]
            
            if grid[new_row][new_col] == expected_char:
                state = (new_row, new_col, next_pattern_idx)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)

print("Yes" if found else "No")