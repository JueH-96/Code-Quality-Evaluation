n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]
result = []

for i in range(n - 8):
    for j in range(m - 8):
        # Check top-left 3x3
        tl_ok = True
        for x in range(i, i+3):
            for y in range(j, j+3):
                if grid[x][y] != '#':
                    tl_ok = False
                    break
            if not tl_ok:
                break
        if not tl_ok:
            continue
        
        # Check bottom-right 3x3
        br_ok = True
        for x in range(i+6, i+9):
            for y in range(j+6, j+9):
                if grid[x][y] != '#':
                    br_ok = False
                    break
            if not br_ok:
                break
        if not br_ok:
            continue
        
        # Collect adjacent cells
        adjacent = []
        # Top-left adjacent
        for y in range(j, j+3):
            adjacent.append((i+3, y))
        for x in range(i, i+3):
            adjacent.append((x, j+3))
        adjacent.append((i+3, j+3))
        
        # Bottom-right adjacent
        for y in range(j+6, j+9):
            adjacent.append((i+5, y))
        for x in range(i+6, i+9):
            adjacent.append((x, j+5))
        adjacent.append((i+5, j+5))
        
        # Check all adjacent are '.'
        adj_ok = True
        for x, y in adjacent:
            if grid[x][y] != '.':
                adj_ok = False
                break
        if adj_ok:
            result.append((i+1, j+1))

# Sort and output
for coord in sorted(result):
    print(coord[0], coord[1])