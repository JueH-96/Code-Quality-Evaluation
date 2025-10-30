def main():
    import sys
    input_lines = sys.stdin.read().splitlines()
    grid = [list(map(int, line.split())) for line in input_lines if line.strip()]
    
    # Check row condition: each row must contain digits 1-9 exactly once.
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            print("No")
            return

    # Check column condition: each column must contain digits 1-9 exactly once.
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if sorted(col) != list(range(1, 10)):
            print("No")
            return

    # Check 3x3 sub-grid condition:
    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = []
            for i in range(block_row, block_row + 3):
                for j in range(block_col, block_col + 3):
                    block.append(grid[i][j])
            if sorted(block) != list(range(1, 10)):
                print("No")
                return
    
    # All conditions satisfied.
    print("Yes")

if __name__ == "__main__":
    main()