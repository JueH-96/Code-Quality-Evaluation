# YOUR CODE HERE
def count_paths(grid, H, W, K):
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def dfs(i, j, steps_remaining, visited):
        if steps_remaining == 0:
            return 1
        
        count = 0
        for di, dj in directions:
            ni, nj = i + di, j + dj
            
            # Check if the new position is valid
            if (0 <= ni < H and 0 <= nj < W and 
                grid[ni][nj] == '.' and (ni, nj) not in visited):
                
                visited.add((ni, nj))
                count += dfs(ni, nj, steps_remaining - 1, visited)
                visited.remove((ni, nj))
        
        return count
    
    total_count = 0
    
    # Try starting from each empty cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = {(i, j)}
                total_count += dfs(i, j, K, visited)
    
    return total_count

# Read input
H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Calculate and print the result
result = count_paths(grid, H, W, K)
print(result)