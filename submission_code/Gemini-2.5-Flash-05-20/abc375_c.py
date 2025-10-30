import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Read the initial grid A_orig
    # A_orig[r][c] will store the character for cell (r, c)
    A_orig = []
    for _ in range(N):
        A_orig.append(sys.stdin.readline().strip())

    # Initialize the result grid B. This grid will store the final state.
    # We will populate B by determining the original source for each final cell.
    B = [['' for _ in range(N)] for _ in range(N)]

    # Iterate through each cell (r_final, c_final) in the final grid B
    for r_final in range(N):
        for c_final in range(N):
            # (r_final, c_final) is the coordinate of the cell in the final grid
            # whose value we want to determine.

            # We trace back the origin of the value at (r_final, c_final)
            # 'current_r' and 'current_c' will hold the coordinates as they
            # evolve through the inverse transformations.
            current_r, current_c = r_final, c_final

            # Determine the 'layer' index k_0 for the cell (r_final, c_final).
            # This k_0 is the smallest k (0-indexed step) such that (r_final, c_final)
            # is part of the k-th square ([k .. N-1-k] x [k .. N-1-k]).
            # Cells in this layer and deeper are affected by operations from step k_0 onwards.
            k_0 = min(current_r, N - 1 - current_r, current_c, N - 1 - current_c)

            # Apply inverse transformations backwards from the last operation (N/2 - 1)
            # down to the first operation that could affect this cell (k_0).
            # The range for k goes from (N/2 - 1) down to k_0.
            # In 0-indexed terms, the problem specifies operations for i=1 to N/2,
            # which corresponds to k=0 to N/2 - 1.
            for k_step in range(N // 2 - 1, k_0 - 1, -1):
                # Check if (current_r, current_c) would have been within the k_step-th square
                # in the grid state before operation k_step.
                # The condition for being in the k_step-th square is:
                # k_step <= row_coord <= N - 1 - k_step
                # AND
                # k_step <= col_coord <= N - 1 - k_step
                if (k_step <= current_r <= N - 1 - k_step and
                    k_step <= current_c <= N - 1 - k_step):
                    
                    # If it was within the square, its value must have come from
                    # an inverse rotation. The forward rotation is (x,y) -> (y, N-1-x).
                    # The inverse rotation (where the value at (current_r, current_c) came FROM)
                    # is (current_r, current_c) = (N-1-current_c, current_r).
                    current_r, current_c = N - 1 - current_c, current_r
                # If (current_r, current_c) was outside the k_step-th square,
                # operation k_step did not affect this position. Its 'ancestor'
                # coordinates remain unchanged for this step.

            # After all relevant inverse transformations, (current_r, current_c)
            # now holds the original coordinates (r_orig, c_orig) from A_orig
            # that ultimately provided the value for B[r_final][c_final].
            B[r_final][c_final] = A_orig[current_r][current_c]

    # Print the resulting grid B
    for r in range(N):
        sys.stdout.write("".join(B[r]) + "
")

# Call the solve function to run the program
solve()