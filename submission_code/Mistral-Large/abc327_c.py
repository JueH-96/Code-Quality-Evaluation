import sys

def is_valid_sudoku(grid):
    # Check each row
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False

    # Check each column
    for col in range(9):
        if sorted(grid[row][col] for row in range(9)) != list(range(1, 10)):
            return False

    # Check each 3x3 sub-grid
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            if sorted(grid[row][col] for row in range(box_row, box_row + 3) for col in range(box_col, box_col + 3)) != list(range(1, 10)):
                return False

    return True

def main():
    input = sys.stdin.read
    data = input().split()

    grid = []
    for i in range(9):
        row = list(map(int, data[i*9:(i+1)*9]))
        grid.append(row)

    if is_valid_sudoku(grid):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()