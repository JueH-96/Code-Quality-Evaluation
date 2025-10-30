n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

results = []

for i in range(n - 9 + 1):
    for j in range(m - 9 + 1):
        valid = True
        
        # Check top-left 3x3
        for x in range(i, i + 3):
            if not valid:
                break
            for y in range(j, j + 3):
                if grid[x][y] != '#':
                    valid = False
                    break
        if not valid:
            continue
        
        # Check bottom-right 3x3
        for x in range(i + 6, i + 9):
            if not valid:
                break
            for y in range(j + 6, j + 9):
                if grid[x][y] != '#':
                    valid = False
                    break
        if not valid:
            continue
        
        # Check top-left adjacent area
        valid_tl = True
        for x in range(i, i + 4):
            if not valid_tl:
                break
            for y in range(j, j + 4):
                if x <= i + 2 and y <= j + 2:
                    continue
                if grid[x][y] != '.':
                    valid_tl = False
                    break
        if not valid_tl:
            continue
        
        # Check bottom-right adjacent area
        valid_br = True
        for x in range(i + 5, i + 9):
            if not valid_br:
                break
            for y in range(j + 5, j + 9):
                if x >= i + 6 and y >= j + 6:
                    continue
                if grid[x][y] != '.':
                    valid_br = False
                    break
        if valid_br:
            results.append((i + 1, j + 1))

for pair in results:
    print(pair[0], pair[1])