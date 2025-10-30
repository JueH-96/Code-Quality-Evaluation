grid = []
for _ in range(8):
    grid.append(input())

safe_squares = 0
for r in range(8):
    for c in range(8):
        if grid[r][c] == '.':
            is_row_safe = True
            for k in range(8):
                if grid[r][k] == '#':
                    is_row_safe = False
                    break

            is_col_safe = True
            for k in range(8):
                if grid[k][c] == '#':
                    is_col_safe = False
                    break

            if is_row_safe and is_col_safe:
                safe_squares += 1

print(safe_squares)