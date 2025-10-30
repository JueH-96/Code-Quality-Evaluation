def solve():
    n = int(input())
    grid = [[0] * n for _ in range(n)]
    
    center_row = (n + 1) // 2 - 1
    center_col = (n + 1) // 2 - 1
    grid[center_row][center_col] = 'T'
    
    num = 1
    row, col = 0, 0
    
    while num < n * n:
        if grid[row][col] == 0:
            grid[row][col] = num
            num += 1
        
        if row == 0 and col < n - 1:
            col += 1
        elif col == n - 1 and row < n - 1:
            row += 1
        elif row == n - 1 and col > 0:
            col -= 1
        elif col == 0 and row > 1:
            row -= 1
        elif row == 1 and col == 0:
            row = 0
            col = 0
            while grid[row][col] != 0:
                if row == 0 and col < n - 1:
                    col += 1
                elif col == n - 1 and row < n - 1:
                    row += 1
                elif row == n - 1 and col > 0:
                    col -= 1
                elif col == 0 and row > 1:
                    row -= 1
                elif row == 1 and col == 0:
                    row = 0
                    col = 0
                    while grid[row][col] != 0:
                        if row == 0 and col < n - 1:
                            col += 1
                        elif col == n - 1 and row < n - 1:
                            row += 1
                        elif row == n - 1 and col > 0:
                            col -= 1
                        elif col == 0 and row > 1:
                            row -= 1
                else:
                    break
        else:
            break
    
    num = 1
    row, col = 0, 0
    
    while num < n * n:
        if grid[row][col] == 0:
            grid[row][col] = num
            num += 1
        
        if row == 0 and col < n - 1:
            col += 1
        elif col == n - 1 and row < n - 1:
            row += 1
        elif row == n - 1 and col > 0:
            col -= 1
        elif col == 0 and row > 1:
            row -= 1
        elif row == 1 and col == 0:
            row = 0
            col = 0
            while grid[row][col] != 0:
                if row == 0 and col < n - 1:
                    col += 1
                elif col == n - 1 and row < n - 1:
                    row += 1
                elif row == n - 1 and col > 0:
                    col -= 1
                elif col == 0 and row > 1:
                    row -= 1
                else:
                    break
        else:
            break
            
    num = 1
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                grid[i][j] = num
                num += 1
    
    num = 1
    row, col = 0, 0
    
    while num < n * n:
        if grid[row][col] == 0:
            grid[row][col] = num
            num += 1
        
        if row == 0 and col < n - 1:
            col += 1
        elif col == n - 1 and row < n - 1:
            row += 1
        elif row == n - 1 and col > 0:
            col -= 1
        elif col == 0 and row > 1:
            row -= 1
        elif row == 1 and col == 0:
            row = 0
            col = 0
            while grid[row][col] != 0:
                if row == 0 and col < n - 1:
                    col += 1
                elif col == n - 1 and row < n - 1:
                    row += 1
                elif row == n - 1 and col > 0:
                    col -= 1
                elif col == 0 and row > 1:
                    row -= 1
                else:
                    break
        else:
            break
    
    num = 1
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                grid[i][j] = num
                num += 1