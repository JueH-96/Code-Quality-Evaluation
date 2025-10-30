def solve():
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)

    def check_rows(grid):
        for row in grid:
            if set(row) != set(range(1, 10)):
                return False
        return True

    def check_cols(grid):
        for j in range(9):
            col = [grid[i][j] for i in range(9)]
            if set(col) != set(range(1, 10)):
                return False
        return True

    def check_subgrids(grid):
        for i_start in range(0, 9, 3):
            for j_start in range(0, 9, 3):
                subgrid_values = []
                for i in range(i_start, i_start + 3):
                    for j in range(j_start, j_start + 3):
                        subgrid_values.append(grid[i][j])
                if set(subgrid_values) != set(range(1, 10)):
                    return False
        return True

    if check_rows(grid) and check_cols(grid) and check_subgrids(grid):
        print("Yes")
    else:
        print("No")

solve()