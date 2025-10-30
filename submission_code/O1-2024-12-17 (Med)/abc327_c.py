def main():
    # Read 9 lines of input, each line has 9 integers
    grid = [list(map(int, input().split())) for _ in range(9)]
    
    # Check rows
    for row in range(9):
        if set(grid[row]) != set(range(1, 10)):
            print("No")
            return
    
    # Check columns
    for col in range(9):
        column_values = [grid[row][col] for row in range(9)]
        if set(column_values) != set(range(1, 10)):
            print("No")
            return
    
    # Check 3x3 subgrids
    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            block_values = []
            for r in range(3):
                for c in range(3):
                    block_values.append(grid[start_row + r][start_col + c])
            if set(block_values) != set(range(1, 10)):
                print("No")
                return
    
    # If all checks pass
    print("Yes")

# Do not forget to call main to run the solution.
main()