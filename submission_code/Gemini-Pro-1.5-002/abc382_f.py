def solve():
    h, w, n = map(int, input().split())
    bars = []
    for _ in range(n):
        r, c, l = map(int, input().split())
        bars.append([r, c, l])

    grid = [[0] * w for _ in range(h)]
    for i in range(n):
        r, c, l = bars[i]
        for j in range(l):
            grid[r-1][c-1+j] = i + 1
    
    for _ in range(1000):
        moved = False
        for i in range(n):
            r, c, l = bars[i]
            if r == h:
                continue

            can_move = True
            for j in range(l):
                if grid[r][c-1+j] != 0 and grid[r][c-1+j] != i+1:
                    can_move = False
                    break
            
            if can_move:
                moved = True
                for j in range(l):
                    grid[r-1][c-1+j] = 0
                    grid[r][c-1+j] = i + 1
                bars[i][0] += 1

        if not moved:
            break

    for r, _, _ in bars:
        print(r)

solve()