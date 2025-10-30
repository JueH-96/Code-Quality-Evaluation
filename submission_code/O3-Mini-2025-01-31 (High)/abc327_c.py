def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    # Convert all input values to integers and build the 9x9 grid.
    nums = list(map(int, data))
    grid = [nums[i*9:(i+1)*9] for i in range(9)]
    
    required = set(range(1, 10))
    
    # Check the rows.
    for row in grid:
        if set(row) != required:
            print("No")
            return

    # Check the columns.
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if set(col) != required:
            print("No")
            return

    # Check the 3x3 subgrids.
    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            block = []
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    block.append(grid[i][j])
            if set(block) != required:
                print("No")
                return

    print("Yes")

if __name__ == '__main__':
    main()