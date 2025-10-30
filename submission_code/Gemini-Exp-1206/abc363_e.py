def solve():
    h, w, y = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))

    for year in range(1, y + 1):
        sunk = [[False] * w for _ in range(h)]
        
        def sink(r, c):
            if not (0 <= r < h and 0 <= c < w):
                return
            if sunk[r][c]:
                return
            if a[r][c] <= year:
                sunk[r][c] = True
                sink(r + 1, c)
                sink(r - 1, c)
                sink(r, c + 1)
                sink(r, c - 1)

        for r in range(h):
            if a[r][0] <= year:
                sink(r, 0)
            if a[r][w - 1] <= year:
                sink(r, w - 1)
        for c in range(w):
            if a[0][c] <= year:
                sink(0, c)
            if a[h - 1][c] <= year:
                sink(h - 1, c)

        count = 0
        for r in range(h):
            for c in range(w):
                if not sunk[r][c]:
                    count += 1
        print(count)

solve()