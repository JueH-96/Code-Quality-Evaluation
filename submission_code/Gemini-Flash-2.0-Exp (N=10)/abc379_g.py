def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    
    q_count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '?':
                q_count += 1
    
    ans = 0
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
    
    def check_grid(current_grid):
        for r in range(h):
            for c in range(w):
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc):
                        if current_grid[r][c] == current_grid[nr][nc]:
                            return False
        return True
    
    def generate_grids(index, current_grid):
        nonlocal ans
        if index == h * w:
            if check_grid(current_grid):
                ans = (ans + 1) % 998244353
            return
        
        r = index // w
        c = index % w
        
        if grid[r][c] != '?':
            generate_grids(index + 1, current_grid)
        else:
            for val in ['1', '2', '3']:
                current_grid[r][c] = val
                generate_grids(index + 1, current_grid)
                current_grid[r][c] = '?'
    
    generate_grids(0, [row[:] for row in grid])
    print(ans)

solve()