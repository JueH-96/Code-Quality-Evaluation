def count_paths(grid, H, W, K):
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(i, j):
        return 0 <= i < H and 0 <= j < W and grid[i][j] == '.'
    
    def dfs(i, j, visited, moves_left):
        if moves_left == 0:
            return 1
        
        count = 0
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if is_valid(ni, nj) and (ni, nj) not in visited:
                visited.add((ni, nj))
                count += dfs(ni, nj, visited, moves_left - 1)
                visited.remove((ni, nj))
        
        return count
    
    total_paths = 0
    # Try starting from each empty cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = {(i, j)}
                total_paths += dfs(i, j, visited, K)
    
    return total_paths

# Read input
H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Calculate and print the result
result = count_paths(grid, H, W, K)
print(result)