def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    results = []
    
    for r in range(n - 8):
        for c in range(m - 8):
            is_tak_code = True
            
            # Check top-left 3x3
            for i in range(r, r + 3):
                for j in range(c, c + 3):
                    if grid[i][j] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
            
            # Check bottom-right 3x3
            for i in range(r + 6, r + 9):
                for j in range(c + 6, c + 9):
                    if grid[i][j] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
            
            # Check adjacent cells to top-left 3x3
            for i in range(r - 1, r + 4):
                for j in range(c - 1, c + 4):
                    if 0 <= i < n and 0 <= j < m and (i < r or i >= r + 3 or j < c or j >= c + 3):
                        if (i == r -1 or i == r+3) and (j >= c-1 and j <= c+3):
                            if grid[i][j] == '#':
                                is_tak_code = False
                                break
                        elif (j == c -1 or j == c+3) and (i >= r-1 and i <= r+3):
                            if grid[i][j] == '#':
                                is_tak_code = False
                                break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
            
            # Check adjacent cells to bottom-right 3x3
            for i in range(r + 5, r + 10):
                for j in range(c + 5, c + 10):
                    if 0 <= i < n and 0 <= j < m and (i < r + 6 or i >= r + 9 or j < c + 6 or j >= c + 9):
                        if (i == r+5 or i == r+9) and (j >= c+5 and j <= c+9):
                            if grid[i][j] == '#':
                                is_tak_code = False
                                break
                        elif (j == c+5 or j == c+9) and (i >= r+5 and i <= r+9):
                            if grid[i][j] == '#':
                                is_tak_code = False
                                break
                if not is_tak_code:
                    break
            if is_tak_code:
                results.append((r + 1, c + 1))
    
    for r, c in results:
        print(r, c)

solve()