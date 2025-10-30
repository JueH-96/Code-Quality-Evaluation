def main():
    import sys

    # Read the 9x9 grid
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(9)]
    
    # Check each row
    for row in grid:
        if set(row) != set(range(1, 10)):
            print("No")
            return
    
    # Check each column
    for col_idx in range(9):
        col = [grid[row_idx][col_idx] for row_idx in range(9)]
        if set(col) != set(range(1, 10)):
            print("No")
            return
    
    # Check each 3x3 sub-grid
    for row_start in range(0, 9, 3):
        for col_start in range(0, 9, 3):
            block = []
            for r in range(3):
                for c in range(3):
                    block.append(grid[row_start + r][col_start + c])
            if set(block) != set(range(1, 10)):
                print("No")
                return
    
    # If all checks pass, print Yes
    print("Yes")

# Do not remove or rename this call
main()