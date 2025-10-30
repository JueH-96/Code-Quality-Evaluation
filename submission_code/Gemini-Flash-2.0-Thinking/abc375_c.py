def solve():
    n = int(input())
    grid = [list(input()) for _ in range(n)]

    for i_op in range(1, n // 2 + 1):
        temp_grid = [row[:] for row in grid]
        for x in range(i_op, n + 2 - i_op):
            for y in range(i_op, n + 2 - i_op):
                grid[y - 1][n - x] = temp_grid[x - 1][y - 1]

    for row in grid:
        print("".join(row))

if __name__ == "__main__":
    solve()