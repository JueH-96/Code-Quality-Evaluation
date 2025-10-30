# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    tak_codes = []
    
    for i in range(n - 8):
        for j in range(m - 8):
            is_tak_code = True
            
            # Check top-left and bottom-right 3x3 regions
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    if grid[row][col] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
                
            for row in range(i + 6, i + 9):
                for col in range(j + 6, j + 9):
                    if grid[row][col] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
            
            # Check adjacent cells
            adjacent_cells = []
            for row in range(i - 1, i + 4):
                if 0 <= row < n:
                    for col in range(j-1, j+4):
                        if 0 <= col < m:
                            adjacent_cells.append((row,col))

            for row in range(i + 5, i + 10):
                if 0 <= row < n:
                    for col in range(j+5, j+10):
                        if 0 <= col < m:
                            adjacent_cells.append((row,col))
            
            
            
            adjacent_white = True
            for row in range(i-1, i+4):
                if 0 <= row < n:
                    if j-1 >= 0 and grid[row][j-1] != '.':
                        adjacent_white = False
                        break
                    if j+3 < m and grid[row][j+3] != '.':
                        adjacent_white = False
                        break

                if not adjacent_white:
                    break
            if not adjacent_white:
                continue
            
            for col in range(j, j+3):
                if i-1 >= 0 and grid[i-1][col] != '.':
                    adjacent_white = False
                    break
                if i+3 < n and grid[i+3][col] != '.':
                    adjacent_white = False
                    break
            if not adjacent_white:
                continue
            
            for row in range(i+5, i+10):
                if 0 <= row < n:
                    if j+5 < m and grid[row][j+10] != '.':
                        adjacent_white = False
                        break
                    if j+5 < m and grid[row][j+5] != '.':
                        adjacent_white = False
                        break

                if not adjacent_white:
                    break
            if not adjacent_white:
                continue
            
            for col in range(j+6, j+9):
                if i+5 < n and grid[i+5][col] != '.':
                    adjacent_white = False
                    break
                if i+9 < n and grid[i+9][col] != '.':
                    adjacent_white = False
                    break
            if not adjacent_white:
                continue
            
            if is_tak_code and adjacent_white:
                tak_codes.append((i + 1, j + 1))

    for i, j in tak_codes:
        print(i, j)

solve()