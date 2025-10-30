# YOUR CODE HERE
from collections import deque

def count_reachable_cells(grid, start_row, start_col):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    count = 0
    queue = deque([(start_row, start_col)])
    
    while queue:
        row, col = queue.popleft()
        if visited[row][col]:
            continue
        
        visited[row][col] = True
        count += 1
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] != '#':
                queue.append((new_row, new_col))
    
    return count

def has_adjacent_magnet(grid, row, col):
    H, W = len(grid), len(grid[0])
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] == '#':
            return True
    return False

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

max_freedom = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '.' and not has_adjacent_magnet(grid, i, j):
            freedom = count_reachable_cells(grid, i, j)
            max_freedom = max(max_freedom, freedom)

print(max_freedom)