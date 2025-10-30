import sys

def solve():
    h, w = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(h)]
    
    red_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                red_cells.append((r, c))

    def count_connected_components(g):
        visited = set()
        count = 0
        
        def dfs(r, c):
            if (r, c) in visited or not (0 <= r < h and 0 <= c < w) or g[r][c] == '.':
                return
            visited.add((r, c))
            dfs(r+1, c)
            dfs(r-1, c)
            dfs(r, c+1)
            dfs(r, c-1)

        for r in range(h):
            for c in range(w):
                if g[r][c] == '#' and (r,c) not in visited:
                    dfs(r,c)
                    count += 1
        return count

    total_components = 0
    for r, c in red_cells:
        temp_grid = [row[:] for row in grid]
        temp_grid[r][c] = '#'
        total_components += count_connected_components(temp_grid)

    mod = 998244353
    
    if len(red_cells) == 0:
        print(count_connected_components(grid))
        return

    inv = pow(len(red_cells), mod - 2, mod)
    print((total_components * inv) % mod)

solve()