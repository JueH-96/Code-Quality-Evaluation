def solve():
    n = int(input())
    size = 3**n
    grid = [['#'] * size for _ in range(size)]

    def carpet(k, row_start, col_start):
        if k == 0:
            return
        
        block_size = 3**(k-1)
        
        # Make the center block white
        for i in range(row_start + block_size, row_start + 2 * block_size):
            for j in range(col_start + block_size, col_start + 2 * block_size):
                grid[i][j] = '.'
        
        # Recurse on the other 8 blocks
        carpet(k-1, row_start, col_start)
        carpet(k-1, row_start, col_start + 2 * block_size)
        carpet(k-1, row_start + 2 * block_size, col_start)
        carpet(k-1, row_start + 2 * block_size, col_start + 2 * block_size)
        
        carpet(k-1, row_start, col_start + block_size)
        carpet(k-1, row_start + block_size, col_start)
        carpet(k-1, row_start + block_size, col_start + 2 * block_size)
        carpet(k-1, row_start + 2 * block_size, col_start + block_size)

    carpet(n, 0, 0)

    for row in grid:
        print("".join(row))

solve()