def count_paths(grid, H, W, K):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    def is_valid(i, j):
        return 0 <= i < H and 0 <= j < W and grid[i][j] == '.'
    
    def dfs(i, j, moves_left, visited):
        if moves_left == 0:
            return 1
        
        count = 0
        visited.add((i, j))
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if is_valid(ni, nj) and (ni, nj) not in visited:
                count += dfs(ni, nj, moves_left - 1, visited)
        
        visited.remove((i, j))
        return count
    
    total_paths = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_paths += dfs(i, j, K, set())
    
    return total_paths

# Read input
H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    row = input().strip()
    grid.append(row)

print(count_paths(grid, H, W, K))