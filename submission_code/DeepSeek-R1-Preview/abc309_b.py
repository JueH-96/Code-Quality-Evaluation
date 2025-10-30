n = int(input())
grid = []
for _ in range(n):
    line = input().strip()
    grid.append([int(c) for c in line])

perimeter_order = []
# Collect top row
for j in range(n):
    perimeter_order.append((0, j))
# Collect right column
for i in range(1, n):
    perimeter_order.append((i, n - 1))
# Collect bottom row
for j in range(n - 2, -1, -1):
    perimeter_order.append((n - 1, j))
# Collect left column
for i in range(n - 2, 0, -1):
    perimeter_order.append((i, 0))

if perimeter_order:
    values = [grid[i][j] for i, j in perimeter_order]
    rotated_values = [values[-1]] + values[:-1]
    for idx, (i, j) in enumerate(perimeter_order):
        grid[i][j] = rotated_values[idx]

for row in grid:
    print(''.join(map(str, row)))