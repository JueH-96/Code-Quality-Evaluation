def solve():
    grid = []
    for _ in range(9):
        grid.append(list(map(int, input().split())))

    def check_rows():
        for row in grid:
            if len(set(row)) != 9 or any(x < 1 or x > 9 for x in row):
                return False
        return True

    def check_cols():
        for j in range(9):
            col = [grid[i][j] for i in range(9)]
            if len(set(col)) != 9 or any(x < 1 or x > 9 for x in col):
                return False
        return True

    def check_subgrids():
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = []
                for row in range(i, i + 3):
                    for col in range(j, j + 3):
                        subgrid.append(grid[row][col])
                if len(set(subgrid)) != 9 or any(x < 1 or x > 9 for x in subgrid):
                    return False
        return True

    if check_rows() and check_cols() and check_subgrids():
        print("Yes")
    else:
        print("No")

solve()