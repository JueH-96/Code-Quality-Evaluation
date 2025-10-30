def main():
    import sys
    input = sys.stdin.read
    lines = input().strip().split('
')
    
    # Read the grid
    grid = []
    for line in lines:
        row = list(map(int, line.strip().split()))
        if len(row) != 9:
            print("No")
            return
        grid.append(row)
    
    if len(grid) != 9:
        print("No")
        return
    
    # Check rows
    for row in grid:
        if set(row) != set(range(1, 10)):
            print("No")
            return
    
    # Check columns
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if set(col) != set(range(1, 10)):
            print("No")
            return
    
    # Check 3x3 subgrids
    for row_start in range(0, 9, 3):
        for col_start in range(0, 9, 3):
            subgrid = []
            for i in range(row_start, row_start + 3):
                for j in range(col_start, col_start + 3):
                    subgrid.append(grid[i][j])
            if set(subgrid) != set(range(1, 10)):
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()