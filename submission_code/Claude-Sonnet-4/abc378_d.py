def solve():
    H, W, K = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W and grid[r][c] == '.'
    
    def dfs(r, c, steps, visited):
        if steps == K:
            return 1
        
        count = 0
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                count += dfs(nr, nc, steps + 1, visited)
                visited.remove((nr, nc))
        
        return count
    
    total_count = 0
    
    # Try starting from each empty cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = {(i, j)}
                total_count += dfs(i, j, 0, visited)
    
    print(total_count)

solve()