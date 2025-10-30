def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    results = []
    
    for i in range(n - 8):
        for j in range(m - 8):
            is_tak_code = True
            
            # Check top-left 3x3 region
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    if grid[row][col] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            
            # Check bottom-right 3x3 region
            if is_tak_code:
                for row in range(i + 6, i + 9):
                    for col in range(j + 6, j + 9):
                        if grid[row][col] != '#':
                            is_tak_code = False
                            break
                    if not is_tak_code:
                        break
            
            # Check adjacent cells to top-left 3x3 region
            if is_tak_code:
                for row_offset in range(-1, 4):
                    for col_offset in range(-1, 4):
                        if 0 <= row_offset < 3 and 0 <= col_offset < 3:
                            continue
                        row = i + row_offset
                        col = j + col_offset
                        if 0 <= row < n and 0 <= col < m:
                            if grid[row][col] != '.':
                                is_tak_code = False
                                break
                    if not is_tak_code:
                        break
            
            # Check adjacent cells to bottom-right 3x3 region
            if is_tak_code:
                for row_offset in range(-1, 4):
                    for col_offset in range(-1, 4):
                        if 0 <= row_offset < 3 and 0 <= col_offset < 3:
                            continue
                        row = i + 6 + row_offset
                        col = j + 6 + col_offset
                        if 0 <= row < n and 0 <= col < m:
                            if grid[row][col] != '.':
                                is_tak_code = False
                                break
                    if not is_tak_code:
                        break
            
            if is_tak_code:
                results.append((i + 1, j + 1))
    
    for i, j in results:
        print(i, j)

solve()