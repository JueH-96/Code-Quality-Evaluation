def main():
    import sys
    
    # Read the input into a 2D list (9x9)
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(9)]
    
    # Check each row
    for row in range(9):
        if set(grid[row]) != set(range(1, 10)):
            print("No")
            return
    
    # Check each column
    for col in range(9):
        column_values = [grid[row][col] for row in range(9)]
        if set(column_values) != set(range(1, 10)):
            print("No")
            return
    
    # Check each 3x3 subgrid
    for sub_row in range(0, 9, 3):
        for sub_col in range(0, 9, 3):
            block_values = []
            for r in range(sub_row, sub_row + 3):
                for c in range(sub_col, sub_col + 3):
                    block_values.append(grid[r][c])
            if set(block_values) != set(range(1, 10)):
                print("No")
                return
    
    # If all checks pass
    print("Yes")

# Do not forget to call main()
main()