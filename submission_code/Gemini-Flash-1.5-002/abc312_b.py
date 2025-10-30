def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    results = []
    for r in range(n - 8):
        for c in range(m - 8):
            region = [grid[r+i][c:c+9] for i in range(9)]
            is_tak_code = True
            
            # Check top-left 3x3
            for i in range(3):
                for j in range(3):
                    if region[i][j] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            
            #Check bottom-right 3x3
            if is_tak_code:
                for i in range(6,9):
                    for j in range(6,9):
                        if region[i][j] != '#':
                            is_tak_code = False
                            break
                    if not is_tak_code:
                        break

            #Check adjacent cells
            if is_tak_code:
                for i in range(3):
                    if region[i][3] != '.' or region[i][4] != '.' or region[i][5] != '.':
                        is_tak_code = False
                        break
                    if region[3][i] != '.' or region[4][i] != '.' or region[5][i] != '.':
                        is_tak_code = False
                        break
                if is_tak_code:
                    for i in range(6,9):
                        if region[i][3] != '.' or region[i][4] != '.' or region[i][5] != '.':
                            is_tak_code = False
                            break
                        if region[3][i] != '.' or region[4][i] != '.' or region[5][i] != '.':
                            is_tak_code = False
                            break
                
                if is_tak_code:
                    if region[3][3] != '.' or region[3][5] != '.' or region[5][3] != '.' or region[5][5] != '.':
                        is_tak_code = False

            if is_tak_code:
                results.append((r + 1, c + 1))

    for r, c in sorted(results):
        print(r, c)

solve()