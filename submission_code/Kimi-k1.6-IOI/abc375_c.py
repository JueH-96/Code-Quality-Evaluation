n = int(input())
original = [list(input().strip()) for _ in range(n)]
result = [[''] * n for _ in range(n)]

for x in range(1, n + 1):
    for y in range(1, n + 1):
        # Compute the layer
        k = min(x-1, n-x, y-1, n-y) + 1
        m = k % 4
        if m == 0:
            nx, ny = x, y
        elif m == 1:
            nx, ny = y, n + 1 - x
        elif m == 2:
            nx, ny = n + 1 - x, n + 1 - y
        else:  # m == 3
            nx, ny = n + 1 - y, x
        # Assign the value to the result grid
        result[nx-1][ny-1] = original[x-1][y-1]

for row in result:
    print(''.join(row))