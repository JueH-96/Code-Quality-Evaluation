def check_grid(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False

    # Check columns
    for col in range(9):
        column = [row[col] for row in grid]
        if sorted(column) != list(range(1, 10)):
            return False

    # Check 3x3 grids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if sorted(subgrid) != list(range(1, 10)):
                return False

    return True

def main():
    grid = [list(map(int, input().split())) for _ in range(9)]
    if check_grid(grid):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()