def dfs(H, W, K, grid, visited, curr_i, curr_j, moves):
    # If we've made K moves, we've found a valid path
    if moves == K:
        return 1
    
    count = 0
    # Try all four directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for di, dj in directions:
        next_i, next_j = curr_i + di, curr_j + dj
        
        # Check if next position is valid
        if (0 <= next_i < H and 
            0 <= next_j < W and 
            grid[next_i][next_j] == '.' and 
            not visited[next_i][next_j]):
            
            # Mark as visited
            visited[next_i][next_j] = True
            # Recurse
            count += dfs(H, W, K, grid, visited, next_i, next_j, moves + 1)
            # Backtrack
            visited[next_i][next_j] = False
            
    return count

def solve():
    # Read input
    H, W, K = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    
    total_paths = 0
    
    # Try starting from each empty cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = [[False] * W for _ in range(H)]
                visited[i][j] = True
                total_paths += dfs(H, W, K, grid, visited, i, j, 0)
    
    print(total_paths)

# Run the solution
solve()