def solve():
    """
    Reads a 9x9 grid and determines if it satisfies Sudoku rules.
    """
    try:
        # Read the 9x9 grid from standard input.
        A = [list(map(int, input().split())) for _ in range(9)]
    except (ValueError, EOFError):
        # This handles cases of malformed or empty input, though problem
        # constraints usually guarantee well-formed input.
        print("No")
        return

    # The required set of numbers for any valid row, column, or box.
    expected_set = set(range(1, 10))

    # Condition 1 & 2: Check all rows and columns.
    for i in range(9):
        # Check row 'i'.
        if set(A[i]) != expected_set:
            print("No")
            return
        
        # Check column 'i' using a set comprehension.
        column_set = {A[j][i] for j in range(9)}
        if column_set != expected_set:
            print("No")
            return

    # Condition 3: Check all 3x3 sub-grids.
    # Iterate through the top-left corner of each 3x3 box.
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            # Use a nested set comprehension for a concise check of the box.
            box_set = {A[r][c] for r in range(i, i + 3) for c in range(j, j + 3)}
            if box_set != expected_set:
                print("No")
                return
    
    # If all checks pass, the grid is valid.
    print("Yes")

# Execute the main logic.
solve()