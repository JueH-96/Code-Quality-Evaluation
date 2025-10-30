n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]
valid = []

for i in range(n - 8):
    for j in range(m - 8):
        # Check top-left 3x3
        tl_ok = True
        for x in range(i, i + 3):
            for y in range(j, j + 3):
                if grid[x][y] != '#':
                    tl_ok = False
                    break
            if not tl_ok:
                break
        if not tl_ok:
            continue
        
        # Check bottom-right 3x3
        br_ok = True
        for x in range(i + 6, i + 9):
            for y in range(j + 6, j + 9):
                if grid[x][y] != '#':
                    br_ok = False
                    break
            if not br_ok:
                break
        if not br_ok:
            continue
        
        # Check adjacent cells
        adj = [
            # Top-left adjacent
            (i, j + 3),
            (i + 1, j + 3),
            (i + 2, j + 3),
            (i + 3, j),
            (i + 3, j + 1),
            (i + 3, j + 2),
            (i + 3, j + 3),
            # Bottom-right adjacent
            (i + 6, j + 5),
            (i + 7, j + 5),
            (i + 8, j + 5),
            (i + 5, j + 6),
            (i + 5, j + 7),
            (i + 5, j + 8),
            (i + 5, j + 5)
        ]
        adj_ok = True
        for x, y in adj:
            if grid[x][y] != '.':
                adj_ok = False
                break
        if adj_ok:
            valid.append((i + 1, j + 1))

for pair in valid:
    print(pair[0], pair[1])