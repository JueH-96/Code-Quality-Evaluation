def solve():
    n = int(input())
    grid = [list(input()) for _ in range(n)]
    
    for i in range(1, n // 2 + 1):
        temp_grid = [row[:] for row in grid]
        for x in range(i, n + 1 - i):
            for y in range(i, n + 1 - i):
                temp_grid[y - 1][n - x] = grid[x - 1][y - 1]
        grid = temp_grid
    
    for row in grid:
        print("".join(row))

solve()