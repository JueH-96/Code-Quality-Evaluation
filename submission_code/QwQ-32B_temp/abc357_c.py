def make_carpet(k):
    if k == 0:
        return ['#']
    else:
        sub = make_carpet(k - 1)
        m = len(sub)
        new_size = 3 * m
        new_grid = [['.' for _ in range(new_size)] for _ in range(new_size)]
        for block_row in range(3):
            for block_col in range(3):
                if block_row == 1 and block_col == 1:
                    continue  # central block remains '.'
                row_start = block_row * m
                col_start = block_col * m
                for i in range(m):
                    for j in range(m):
                        c = sub[i][j]
                        new_grid[row_start + i][col_start + j] = c
        return [''.join(row) for row in new_grid]

n = int(input())
carpet = make_carpet(n)
for line in carpet:
    print(line)