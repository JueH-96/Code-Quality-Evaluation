def main():
    import sys
    input_data = sys.stdin.read().split()
    # There should be exactly 9*9 = 81 integers.
    if len(input_data) != 81:
        return  # Something's wrong with the input.
    
    # Convert input list to 9x9 grid of integers.
    grid = []
    index = 0
    for _ in range(9):
        row = list(map(int, input_data[index:index+9]))
        grid.append(row)
        index += 9

    # Define the set of valid digits.
    valid_set = set(range(1, 10))
    
    # Check each row.
    for row in grid:
        if set(row) != valid_set:
            print("No")
            return

    # Check each column.
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if set(col) != valid_set:
            print("No")
            return

    # Check each 3x3 subgrid.
    for bi in range(0, 9, 3):
        for bj in range(0, 9, 3):
            block = []
            for i in range(bi, bi+3):
                for j in range(bj, bj+3):
                    block.append(grid[i][j])
            if set(block) != valid_set:
                print("No")
                return

    # If all conditions passed
    print("Yes")

if __name__ == '__main__':
    main()