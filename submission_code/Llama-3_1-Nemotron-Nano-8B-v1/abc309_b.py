n = int(input())
grid = [list(input().strip()) for _ in range(n)]

perimeter = []
for j in range(n):
    perimeter.append((0, j))
for i in range(1, n):
    perimeter.append((i, n-1))
for j in range(n-2, -1, -1):
    perimeter.append((n-1, j))
for i in range(n-2, 0, -1):
    perimeter.append((i, 0))

original = [grid[i][j] for i, j in perimeter]
rotated = original[-1:] + original[:-1]

for idx, (i, j) in enumerate(perimeter):
    grid[i][j] = rotated[idx]

for row in grid:
    print(''.join(row))