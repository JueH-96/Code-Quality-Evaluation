def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    cookies = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '#':
                cookies.append((r, c))

    if not cookies:
        for r in range(h):
            for c in range(w):
                if grid[r][c] == '.':
                    print(r + 1, c + 1)
                    return

    min_r = min(r for r, c in cookies)
    max_r = max(r for r, c in cookies)
    min_c = min(c for r, c in cookies)
    max_c = max(c for r, c in cookies)

    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == '.':
                print(r + 1, c + 1)
                return

solve()