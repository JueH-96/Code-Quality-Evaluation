import sys

def solve():
    # Read H and W from the first line of input
    H, W = map(int, sys.stdin.readline().split())

    # Read the grid rows into a list of strings
    # .strip() removes trailing newline characters
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize variables to find the bounding box of existing '#' cells.
    # min_r: minimum row index of a '#' cell. Initialized to H (one past max index).
    # max_r: maximum row index of a '#' cell. Initialized to -1 (one before min index).
    # min_c: minimum column index of a '#' cell. Initialized to W (one past max index).
    # max_c: maximum column index of a '#' cell. Initialized to -1 (one before min index).
    min_r = H
    max_r = -1
    min_c = W
    max_c = -1

    # Step 1: Iterate through the grid to find the exact boundaries (min/max row/column indices)
    # of all existing black cells ('#'). This defines the unique candidate rectangle.
    for r in range(H):
        for c in range(W):
            if S[r][c] == '#':
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    # According to problem constraints: "There is at least one cell that is already painted black."
    # This guarantees that min_r, max_r, min_c, max_c will be updated to valid indices (0 to H-1/W-1)
    # and max_r >= min_r, max_c >= min_c.

    # Step 2: Iterate through the grid again to verify if the determined candidate rectangle
    # is consistent with all existing '.' (white) and '#' (black) cells.
    for r in range(H):
        for c in range(W):
            # Check if the current cell (r, c) falls within the candidate rectangle
            is_inside_candidate_rect = (min_r <= r <= max_r and min_c <= c <= max_c)

            if is_inside_candidate_rect:
                # If a cell is inside the candidate rectangle, it must eventually be black.
                # So, if it's currently '.', it's an impossible scenario.
                if S[r][c] == '.':
                    print("No")
                    return # Exit the function immediately
            else:
                # If a cell is outside the candidate rectangle, it must eventually be white.
                # So, if it's currently '#', it's an impossible scenario.
                if S[r][c] == '#':
                    print("No")
                    return # Exit the function immediately
    
    # If the program reaches this point, it means all checks passed.
    # A valid completion is possible by painting '?' cells accordingly.
    print("Yes")

# Call the solve function to execute the program
solve()