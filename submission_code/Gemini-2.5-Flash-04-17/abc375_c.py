import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read initial grid
    # Use list of lists for mutability
    initial_grid = [list(sys.stdin.readline().strip()) for _ in range(N)]

    # Create the final grid
    final_grid = [['' for _ in range(N)] for _ in range(N)]

    # Iterate through each cell (r_target, c_target) in the final grid
    # We want to find which initial cell's color ends up at (r_target, c_target)
    for r_target in range(N):
        for c_target in range(N):
            # The current position we are tracking backward from
            curr_r, curr_c = r_target, c_target

            # Iterate backward through the operations
            # Operations are performed for i = 1, 2, ..., N/2
            # In 0-based index, i_0 = 0, 1, ..., N/2 - 1
            # We go backward from i_0 = N/2 - 1 down to 0
            for i_0 in range(N // 2 - 1, -1, -1):
                # The square region for operation i_0 is [i_0, N - 1 - i_0] x [i_0, N - 1 - i_0] (0-based)
                start = i_0
                end = N - 1 - i_0

                # Check if the current position (curr_r, curr_c) was within the square S_{i_0}
                # in the grid state *after* step i_0, which is the state we are tracing back *from*.
                # If the cell (curr_r, curr_c) in Grid_{i_0+1} is within S_{i_0}, it must have been a target cell
                # of the operation in step i_0.
                if start <= curr_r <= end and start <= curr_c <= end:
                    # The forward operation maps (x_0, y_0) -> (y_0, N - 1 - x_0).
                    # If (curr_r, curr_c) in Grid_{i_0+1} was the target (y_0, N - 1 - x_0),
                    # then the source (x_0, y_0) in Grid_{i_0} was (N - 1 - curr_c, curr_r).
                    # Apply the inverse transformation to find the source position in the previous grid state.
                    temp_r = curr_r
                    curr_r = N - 1 - curr_c
                    curr_c = temp_r
                # Else: The current position (curr_r, curr_c) was outside the square S_{i_0}
                # in the grid state *after* step i_0.
                # Cells outside S_{i_0} are not updated by the operation in step i_0.
                # Therefore, the color at (curr_r, curr_c) in Grid_{i_0+1} must have been the
                # same as the color at (curr_r, curr_c) in Grid_{i_0}.
                # The position we need to track back in Grid_{i_0} is still (curr_r, curr_c).

            # After iterating through all operations backward (from N/2-1 down to 0),
            # the final (curr_r, curr_c) is the original position in Grid_0
            # whose color ends up at (r_target, c_target) in the final grid.
            final_grid[r_target][c_target] = initial_grid[curr_r][curr_c]

    # Print the final grid
    for row in final_grid:
        print("".join(row))

solve()