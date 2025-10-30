import sys

def solve():
    # Read H and W
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid into a list of strings
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Find the bounding box of existing black cells '#'.
    # Initialize min/max row/column indices. Use values that will be updated
    # by any valid 0-indexed row/column (0 to H-1 for rows, 0 to W-1 for columns).
    min_r, max_r = H, -1  # Initialize min_r to H (larger than any valid index), max_r to -1 (smaller than any valid index)
    min_c, max_c = W, -1  # Initialize min_c to W, max_c to -1

    # Iterate through the grid to find the minimum and maximum row and column
    # indices where a '#' character is present.
    # The problem guarantees at least one '#' exists, so min_r, max_r, min_c, max_c
    # will be updated from their initial states to represent a valid, non-empty box
    # after this loop.
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)

    # For all black cells to form a rectangle, this rectangle must necessarily
    # be the minimal bounding box that contains all the initially existing '#' cells.
    # Let this proposed black rectangle be defined by the rows from min_r to max_r
    # and columns from min_c to max_c (using 0-based indexing).
    # We now need to check if the existing '.' and '#' cells in the grid are
    # consistent with this specific proposed target rectangle.

    # Iterate through every cell in the grid again to perform the consistency checks.
    for i in range(H):
        for j in range(W):
            # Determine if the current cell (i, j) falls within the boundaries
            # of the proposed black rectangle.
            is_inside_proposed_rectangle = (min_r <= i <= max_r) and (min_c <= j <= max_c)

            if is_inside_proposed_rectangle:
                # If a cell is located inside the required black rectangle, it must
                # ultimately be black in the final configuration.
                # This means it can either be already painted black ('#') or be unpainted ('?'),
                # which can then be painted black.
                # If this cell is already painted white ('.') in the input grid,
                # it contradicts the requirement for cells inside the rectangle.
                # Therefore, it's impossible to form the rectangle.
                if grid[i][j] == '.':
                    print("No")
                    return # Contradiction found, exit the function immediately.

            else: # The cell (i, j) is outside the proposed black rectangle.
                # If a cell is located outside the required black rectangle, it must
                # ultimately be white in the final configuration.
                # This means it can either be already painted white ('.') or be unpainted ('?'),
                # which can then be painted white.
                # If this cell is already painted black ('#') in the input grid,
                # it contradicts the requirement for cells outside the rectangle.
                # Therefore, it's impossible to form the rectangle.
                if grid[i][j] == '#':
                    print("No")
                    return # Contradiction found, exit the function immediately.

    # If the code reaches this point, it means that no contradictions were found
    # during the checks. All existing '.' cells are outside the proposed rectangle,
    # and all existing '#' cells are inside the proposed rectangle.
    # In this scenario, it is possible to paint the '?' cells to form the required
    # black rectangle:
    # - Paint all '?' cells located inside the proposed bounding box as '#'.
    # - Paint all '?' cells located outside the proposed bounding box as '.'.
    # This results in all black cells forming exactly the rectangle defined by
    # the bounding box [min_r:max_r+1, min_c:max_c+1].
    print("Yes")

# Call the main function to execute the logic.
solve()