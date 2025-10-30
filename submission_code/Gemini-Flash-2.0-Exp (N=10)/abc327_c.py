def solve():
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)

    # Check rows
    for row in grid:
        if len(set(row)) != 9 or any(x < 1 or x > 9 for x in row):
            print("No")
            return

    # Check columns
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if len(set(col)) != 9 or any(x < 1 or x > 9 for x in col):
            print("No")
            return

    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = []
            for row_offset in range(3):
                for col_offset in range(3):
                    subgrid.append(grid[i + row_offset][j + col_offset])
            if len(set(subgrid)) != 9 or any(x < 1 or x > 9 for x in subgrid):
                print("No")
                return

    print("Yes")

solve()