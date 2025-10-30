def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    ans = 0
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w and grid[r][c] == '.'

    def count_paths(path):
        nonlocal ans
        if len(path) == k + 1:
            ans += 1
            return

        last_r, last_c = path[-1]
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = last_r + dr, last_c + dc
            if is_valid(nr, nc) and (nr, nc) not in path:
                count_paths(path + [(nr, nc)])

    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                count_paths([(r, c)])
    
    print(ans)

solve()