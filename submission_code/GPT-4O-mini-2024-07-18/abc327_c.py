def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False

    # Check columns
    for col in range(9):
        if sorted(grid[row][col] for row in range(9)) != list(range(1, 10)):
            return False

    # Check 3x3 subgrids
    for box_row in range(3):
        for box_col in range(3):
            subgrid = []
            for i in range(3):
                for j in range(3):
                    subgrid.append(grid[box_row * 3 + i][box_col * 3 + j])
            if sorted(subgrid) != list(range(1, 10)):
                return False

    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().splitlines()
    
    grid = [list(map(int, line.split())) for line in data]
    
    if is_valid_sudoku(grid):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()