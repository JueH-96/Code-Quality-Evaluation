def solve():
    n = int(input())
    size = 3**n
    grid = [['#' for _ in range(size)] for _ in range(size)]

    def carpet(level, row_start, col_start):
        if level == 0:
            return
        
        sub_size = 3**(level - 1)
        
        for i in range(row_start + sub_size, row_start + 2 * sub_size):
            for j in range(col_start + sub_size, col_start + 2 * sub_size):
                grid[i][j] = '.'
        
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                carpet(level - 1, row_start + i * sub_size, col_start + j * sub_size)

    carpet(n, 0, 0)
    
    for row in grid:
        print("".join(row))

solve()