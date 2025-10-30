# YOUR CODE HERE
def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if set(row) != set(range(1, 10)):
            return False
    
    # Check columns
    for col in range(9):
        if set(grid[row][col] for row in range(9)) != set(range(1, 10)):
            return False
    
    # Check 3x3 sub-grids
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            sub_grid = []
            for i in range(3):
                for j in range(3):
                    sub_grid.append(grid[box_row + i][box_col + j])
            if set(sub_grid) != set(range(1, 10)):
                return False
    
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    # Convert input to a 9x9 grid of integers
    grid = []
    for i in range(9):
        grid.append(list(map(int, data[i*9:(i+1)*9])))
    
    if is_valid_sudoku(grid):
        print("Yes")
    else:
        print("No")