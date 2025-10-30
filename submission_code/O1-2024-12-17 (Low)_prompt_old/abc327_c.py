def solve():
    import sys
    
    # Read the 9x9 grid
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(9)]
    
    # Check rows
    for row in range(9):
        if len(set(grid[row])) != 9:
            print("No")
            return
    
    # Check columns
    for col in range(9):
        column_nums = [grid[row][col] for row in range(9)]
        if len(set(column_nums)) != 9:
            print("No")
            return
    
    # Check 3x3 subgrids
    for sub_row in range(0, 9, 3):
        for sub_col in range(0, 9, 3):
            subgrid_nums = []
            for r in range(sub_row, sub_row + 3):
                for c in range(sub_col, sub_col + 3):
                    subgrid_nums.append(grid[r][c])
            if len(set(subgrid_nums)) != 9:
                print("No")
                return
    
    # If all checks pass
    print("Yes")

def main():
    solve()

# If you want to run the main function, uncomment the following line:
# main()