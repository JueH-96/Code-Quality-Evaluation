import sys

def shift_outer_squares(grid, n):
    # Extract the outer squares in clockwise order
    top_row = grid[0][:]
    right_col = [grid[i][n-1] for i in range(1, n-1)]
    bottom_row = grid[n-1][:]
    left_col = [grid[i][0] for i in range(1, n-1)]
    
    # Rotate the extracted squares clockwise
    rotated_top_row = [bottom_row[0]] + top_row[:-1]
    rotated_right_col = [bottom_row[-1]] + right_col[:-1]
    rotated_bottom_row = [left_col[-1]] + bottom_row[:-1]
    rotated_left_col = [top_row[0]] + left_col[:-1]
    
    # Update the grid with the rotated values
    grid[0] = rotated_top_row
    for i in range(1, n-1):
        grid[i][n-1] = rotated_right_col[i-1]
    grid[n-1] = rotated_bottom_row
    for i in range(1, n-1):
        grid[i][0] = rotated_left_col[i-1]
    
    return grid

def main():
    input_lines = sys.stdin.read().splitlines()
    n = int(input_lines[0])
    grid = [list(map(int, list(line))) for line in input_lines[1:]]
    
    shifted_grid = shift_outer_squares(grid, n)
    
    for row in shifted_grid:
        print(''.join(map(str, row)))

if __name__ == "__main__":
    main()