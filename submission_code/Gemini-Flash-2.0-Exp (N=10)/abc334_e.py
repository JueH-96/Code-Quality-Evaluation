def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    red_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                red_cells.append((r, c))
    
    num_red_cells = len(red_cells)
    
    total_components = 0
    
    def count_components(temp_grid):
        visited = [[False for _ in range(w)] for _ in range(h)]
        count = 0
        
        def dfs(r, c):
            if r < 0 or r >= h or c < 0 or c >= w or visited[r][c] or temp_grid[r][c] == '.':
                return
            visited[r][c] = True
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
            
        for r in range(h):
            for c in range(w):
                if temp_grid[r][c] == '#' and not visited[r][c]:
                    dfs(r, c)
                    count += 1
        return count
    
    for r, c in red_cells:
        temp_grid = [list(row) for row in grid]
        temp_grid[r][c] = '#'
        temp_grid = ["".join(row) for row in temp_grid]
        total_components += count_components(temp_grid)
        
    
    if num_red_cells == 0:
        print(0)
        return

    
    p = total_components
    q = num_red_cells
    
    mod = 998244353
    
    def power(base, exp, m):
        res = 1
        base %= m
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % m
            exp >>= 1
            base = (base * base) % m
        return res
    
    def mod_inverse(a, m):
        return power(a, m - 2, m)
    
    
    result = (p * mod_inverse(q, mod)) % mod
    print(result)

solve()