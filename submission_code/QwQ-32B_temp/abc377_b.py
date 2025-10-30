grid = [input().strip() for _ in range(8)]

rows_has_piece = ['#' in row for row in grid]

cols_has_piece = [False] * 8
for j in range(8):
    cols_has_piece[j] = any(grid[i][j] == '#' for i in range(8))

count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and not rows_has_piece[i] and not cols_has_piece[j]:
            count += 1

print(count)