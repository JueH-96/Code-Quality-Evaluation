n = int(input())
grid = [['.' for _ in range(n)] for _ in range(n)]

for i in range(1, n + 1):
    j = n + 1 - i
    if i > j:
        continue
    color = '#' if i % 2 == 1 else '.'
    start_row = i - 1
    end_row = j - 1
    start_col = i - 1
    end_col = j - 1
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            grid[row][col] = color

for row in grid:
    print(''.join(row))