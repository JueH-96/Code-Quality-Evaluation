n = int(input())
grid = [['.' for _ in range(n)] for _ in range(n)]

for i in range(1, n + 1):
    j = n + 1 - i
    if i > j:
        continue
    color = '#' if i % 2 == 1 else '.'
    for row in range(i - 1, j):
        for col in range(i - 1, j):
            grid[row][col] = color

for row in grid:
    print(''.join(row))