n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

must_be_black = []
# Top-left 3x3
for x in range(3):
    for y in range(3):
        must_be_black.append((x, y))
# Bottom-right 3x3
for x in range(6, 9):
    for y in range(6, 9):
        must_be_black.append((x, y))

must_be_white = [
    (0, 3), (1, 3), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3),
    (5, 5), (5, 6), (5, 7), (5, 8),
    (6, 5), (7, 5), (8, 5)
]

result = []

for i in range(1, n - 8 + 1):
    for j in range(1, m - 8 + 1):
        valid = True
        # Check black cells
        for (x, y) in must_be_black:
            if grid[i-1 + x][j-1 + y] != '#':
                valid = False
                break
        if not valid:
            continue
        # Check white cells
        for (x, y) in must_be_white:
            if grid[i-1 + x][j-1 + y] != '.':
                valid = False
                break
        if valid:
            result.append((i, j))

for pair in result:
    print(pair[0], pair[1])