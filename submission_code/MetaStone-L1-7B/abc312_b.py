n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

for i in range(n - 8):
    for j in range(m - 8):
        valid = True
        
        # Check top-left 3x3
        for x in range(i, i + 3):
            for y in range(j, j + 3):
                if grid[x][y] != '#':
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Check bottom-right 3x3
        for x in range(i + 6, i + 9):
            for y in range(j + 6, j + 9):
                if grid[x][y] != '#':
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Check each cell in 9x9
        for x in range(i, i + 9):
            for y in range(j, j + 9):
                # Skip top-left and bottom-right 3x3
                if (x >= i and x <= i + 2) and (y >= j and y <= j + 2):
                    continue
                if (x >= i + 6 and x <= i + 8) and (y >= j + 6 and y <= j + 8):
                    continue
                
                # Check adjacent to top-left 3x3
                adjacent_top = False
                for a in [i, i + 1, i + 2]:
                    for b in [j, j + 1, j + 2]:
                        if abs(x - a) <= 1 and abs(y - b) <= 1 and (x != a or y != b):
                            adjacent_top = True
                            break
                    if adjacent_top:
                        break
                
                if not adjacent_top:
                    # Check adjacent to bottom-right 3x3
                    adjacent_bottom = False
                    for a in [i + 6, i + 7, i + 8]:
                        for b in [j + 6, j + 7, j + 8]:
                            if abs(x - a) <= 1 and abs(y - b) <= 1 and (x != a or y != b):
                                adjacent_bottom = True
                                break
                        if adjacent_bottom:
                            break
                
                # Check cell value based on adjacency
                if adjacent_top:
                    if grid[x][y] != '.':
                        valid = False
                        break
                elif adjacent_bottom:
                    if grid[x][y] != '.':
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            print(i + 1, j + 1)