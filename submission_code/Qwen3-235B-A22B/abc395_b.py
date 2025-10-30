n = int(input())
grid = [['.' for _ in range(n)] for _ in range(n)]

for i in range(1, n + 1):
    j = n + 1 - i
    if i > j:
        continue
    color = '#' if (i % 2 == 1) else '.'
    top = i - 1
    left = i - 1
    bottom = j - 1
    right = j - 1
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            grid[row][col] = color

for line in grid:
    print(''.join(line))