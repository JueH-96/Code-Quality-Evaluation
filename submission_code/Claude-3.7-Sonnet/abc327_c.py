def check_rows(grid):
    for row in grid:
        if set(row) != set(range(1, 10)):
            return False
    return True

def check_columns(grid):
    for j in range(9):
        column = [grid[i][j] for i in range(9)]
        if set(column) != set(range(1, 10)):
            return False
    return True

def check_3x3_grids(grid):
    for start_row in [0, 3, 6]:
        for start_col in [0, 3, 6]:
            subgrid = []
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    subgrid.append(grid[i][j])
            if set(subgrid) != set(range(1, 10)):
                return False
    return True

def main():
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)
    
    if check_rows(grid) and check_columns(grid) and check_3x3_grids(grid):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()