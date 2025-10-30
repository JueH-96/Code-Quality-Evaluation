def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    target = ['s', 'n', 'u', 'k', 'e']

    for r1 in range(h):
        for c1 in range(w):
            if grid[r1][c1] == target[0]:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        r2, c2 = r1 + dr, c1 + dc
                        if 0 <= r2 < h and 0 <= c2 < w and grid[r2][c2] == target[1]:
                            r3, c3 = r2 + dr, c2 + dc
                            if 0 <= r3 < h and 0 <= c3 < w and grid[r3][c3] == target[2]:
                                r4, c4 = r3 + dr, c3 + dc
                                if 0 <= r4 < h and 0 <= c4 < w and grid[r4][c4] == target[3]:
                                    r5, c5 = r4 + dr, c4 + dc
                                    if 0 <= r5 < h and 0 <= c5 < w and grid[r5][c5] == target[4]:
                                        print(r1 + 1, c1 + 1)
                                        print(r2 + 1, c2 + 1)
                                        print(r3 + 1, c3 + 1)
                                        print(r4 + 1, c4 + 1)
                                        print(r5 + 1, c5 + 1)
                                        return

solve()