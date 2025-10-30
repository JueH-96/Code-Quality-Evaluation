# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def count_components(g):
        count = 0
        visited = [[False] * w for _ in range(h)]
        
        def dfs(r, c):
            if r < 0 or r >= h or c < 0 or c >= w or visited[r][c] or g[r][c] == '.':
                return
            visited[r][c] = True
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        for r in range(h):
            for c in range(w):
                if g[r][c] == '#' and not visited[r][c]:
                    dfs(r, c)
                    count += 1
        return count

    red_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                red_cells.append((r, c))
    
    total_components = 0
    for r, c in red_cells:
        new_grid = [list(row) for row in grid]
        new_grid[r][c] = '#'
        total_components = (total_components + count_components(new_grid)) % 998244353
        
    
    num_red_cells = len(red_cells)
    if num_red_cells == 0:
        print(count_components(grid))
        return
    
    inv_red_cells = pow(num_red_cells, 998244353 - 2, 998244353)
    expected_value = (total_components * inv_red_cells) % 998244353
    print(expected_value)

solve()