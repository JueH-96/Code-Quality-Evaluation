# YOUR CODE HERE
from collections import deque

def count_paths(H, W, K, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'
    
    def dfs(x, y, moves, visited):
        if moves == K:
            return 1
        
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                count += dfs(nx, ny, moves + 1, visited)
                visited.remove((nx, ny))
        
        return count
    
    total_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_count += dfs(i, j, 0, {(i, j)})
    
    return total_count

# Read input
H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Solve and print output
result = count_paths(H, W, K, grid)
print(result)