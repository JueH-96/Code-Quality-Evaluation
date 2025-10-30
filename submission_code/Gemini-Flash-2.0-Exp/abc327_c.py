def solve():
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Check rows
    for row in grid:
        if len(set(row)) != 9:
            print("No")
            return
    
    # Check columns
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if len(set(col)) != 9:
            print("No")
            return
    
    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = []
            for row in range(i, i+3):
                for col in range(j, j+3):
                    subgrid.append(grid[row][col])
            if len(set(subgrid)) != 9:
                print("No")
                return
    
    print("Yes")

solve()