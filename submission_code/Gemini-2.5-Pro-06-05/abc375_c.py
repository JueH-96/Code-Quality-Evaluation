import sys

def solve():
    """
    Reads the grid, calculates the final state, and prints it.
    The solution avoids simulating the rotations step-by-step, which would be O(N^3).
    Instead, it calculates the final source for each cell's color in O(N^2).
    """
    try:
        # Read N from stdin. Use sys.stdin.readline for performance.
        n_str = sys.stdin.readline()
        if not n_str:
            return
        N = int(n_str)
        
        # Read the initial grid state.
        initial_grid = [sys.stdin.readline().strip() for _ in range(N)]
    except (IOError, ValueError):
        # Handle potential empty input or parsing errors.
        return

    # Create a grid to store the final state.
    final_grid = [['' for _ in range(N)] for _ in range(N)]

    # For each cell (r, c) in the final grid, determine its color.
    for r in range(N):
        for c in range(N):
            # 1. Calculate the "depth" of the cell (r, c).
            # The depth is the 0-based index of the concentric square shell
            # the cell belongs to, starting from the outside.
            depth = min(r, c, N - 1 - r, N - 1 - c)

            # 2. A cell at depth 'd' is part of d+1 nested subgrids that get
            # rotated. So, its color's position is transformed d+1 times.
            num_rotations = depth + 1

            # 3. The color at a final destination cell `dest` comes from an initial
            # source cell `src`. Since the color at `src` is moved by `k` clockwise
            # rotations, we find `src` by applying `k` counter-clockwise rotations
            # to `dest`.
            # Let `dest = (r, c)`. We find `src = (r_src, c_src)`.
            
            # A 90-degree CCW rotation maps a point (x, y) to (N-1-y, x).
            # We only need the net effect, which depends on num_rotations % 4.
            
            rot_type = num_rotations % 4
            
            r_curr, c_curr = r, c

            if rot_type == 1:  # 1 CCW rotation (90 degrees)
                r_src, c_src = N - 1 - c_curr, r_curr
            elif rot_type == 2:  # 2 CCW rotations (180 degrees)
                r_src, c_src = N - 1 - r_curr, N - 1 - c_curr
            elif rot_type == 3:  # 3 CCW rotations (270 degrees)
                r_src, c_src = c_curr, N - 1 - r_curr
            else:  # 0 or 4 rotations (identity)
                r_src, c_src = r_curr, c_curr
            
            # Assign the color from the source cell in the initial grid.
            final_grid[r][c] = initial_grid[r_src][c_src]

    # Print the resulting grid to stdout.
    for row in final_grid:
        sys.stdout.write("".join(row) + "
")

solve()