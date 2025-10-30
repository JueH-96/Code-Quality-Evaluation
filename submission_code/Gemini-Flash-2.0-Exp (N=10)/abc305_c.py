def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    min_r, max_r = h, -1
    min_c, max_c = w, -1
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '#':
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == '.':
                print(r + 1, c + 1)
                return

solve()