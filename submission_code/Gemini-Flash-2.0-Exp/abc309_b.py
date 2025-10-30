def solve():
    n = int(input())
    grid = []
    for _ in range(n):
        grid.append(list(input()))

    outer_values = []
    for j in range(n):
        outer_values.append(grid[0][j])
    for i in range(1, n):
        outer_values.append(grid[i][n-1])
    for j in range(n-2, -1, -1):
        outer_values.append(grid[n-1][j])
    for i in range(n-2, 0, -1):
        outer_values.append(grid[i][0])

    shifted_outer_values = [outer_values[-1]] + outer_values[:-1]

    k = 0
    for j in range(n):
        grid[0][j] = shifted_outer_values[k]
        k += 1
    for i in range(1, n):
        grid[i][n-1] = shifted_outer_values[k]
        k += 1
    for j in range(n-2, -1, -1):
        grid[n-1][j] = shifted_outer_values[k]
        k += 1
    for i in range(n-2, 0, -1):
        grid[i][0] = shifted_outer_values[k]
        k += 1

    for row in grid:
        print("".join(row))

solve()