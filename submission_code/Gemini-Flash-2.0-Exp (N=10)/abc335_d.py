def solve():
    n = int(input())
    grid = [[0] * n for _ in range(n)]
    center = (n // 2, n // 2)
    grid[center[0]][center[1]] = 'T'
    
    num = 1
    row, col = 0, 0
    
    while num < n * n:
        while col < n and grid[row][col] == 0:
            if grid[row][col] == 0:
                grid[row][col] = num
                num += 1
            col += 1
        col -= 1
        row += 1
        
        while row < n and grid[row][col] == 0:
            if grid[row][col] == 0:
                grid[row][col] = num
                num += 1
            row += 1
        row -= 1
        col -= 1
        
        while col >= 0 and grid[row][col] == 0:
            if grid[row][col] == 0:
                grid[row][col] = num
                num += 1
            col -= 1
        col += 1
        row -= 1
        
        while row >= 0 and grid[row][col] == 0:
            if grid[row][col] == 0:
                grid[row][col] = num
                num += 1
            row -= 1
        row += 1
        col += 1
        
    for row in grid:
        print(*row)

solve()