def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))
    
    num_red_cells = len(red_cells)
    
    total_components = 0
    for r, c in red_cells:
        new_grid = [list(row) for row in grid]
        new_grid[r][c] = '#'
        
        def count_components(grid):
            visited = [[False] * W for _ in range(H)]
            num_components = 0
            
            def dfs(i, j):
                if i < 0 or i >= H or j < 0 or j >= W or visited[i][j] or grid[i][j] == '.':
                    return
                
                visited[i][j] = True
                dfs(i + 1, j)
                dfs(i - 1, j)
                dfs(i, j + 1)
                dfs(i, j - 1)
            
            for i in range(H):
                for j in range(W):
                    if grid[i][j] == '#' and not visited[i][j]:
                        dfs(i, j)
                        num_components += 1
            
            return num_components
        
        total_components += count_components(new_grid)
    
    expected_value = total_components / num_red_cells
    
    # Calculate modular inverse
    def mod_inverse(a, m):
        m0 = m
        y = 0
        x = 1
        
        if m == 1:
            return 0
        
        while a > 1:
            q = a // m
            t = m
            m = a % m
            a = t
            t = y
            y = x - q * y
            x = t
        
        if x < 0:
            x = x + m0
        
        return x
    
    p = total_components
    q = num_red_cells
    
    r = (p * mod_inverse(q, 998244353)) % 998244353
    
    print(int(r))

solve()