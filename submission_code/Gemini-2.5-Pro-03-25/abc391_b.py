# YOUR CODE HERE
import sys

def solve():
    # Read N and M, the dimensions of the grids S and T respectively.
    # N is the size of the larger grid S (N x N).
    # M is the size of the smaller grid T (M x M).
    n, m = map(int, sys.stdin.readline().split())

    # Read the N x N grid S into a list of strings. Each string represents a row.
    grid_s = [sys.stdin.readline().strip() for _ in range(n)]

    # Read the M x M grid T into a list of strings. Each string represents a row.
    grid_t = [sys.stdin.readline().strip() for _ in range(m)]

    # We need to find the top-left corner (a, b) of an M x M subgrid within S that matches T.
    # The coordinates (a, b) are 1-based.
    # We will iterate through all possible top-left corners using 0-based indices (row_start, col_start).
    # The possible range for row_start is 0 to N - M.
    # The possible range for col_start is 0 to N - M.

    # Iterate through all possible starting row indices (0-based) for the subgrid in S.
    for row_start in range(n - m + 1):
        # Iterate through all possible starting column indices (0-based) for the subgrid in S.
        for col_start in range(n - m + 1):
            
            # Assume that the M x M subgrid starting at (row_start, col_start) in S matches T.
            # We will verify this assumption row by row.
            match = True 
            
            # Iterate through the rows of the template grid T (from index 0 to m-1).
            # For each row i in T, compare it with the corresponding row segment in the subgrid of S.
            for i in range(m):
                # Calculate the row index in the larger grid S corresponding to the i-th row of the subgrid.
                s_row_index = row_start + i
                
                # Extract the substring from grid_s that corresponds to the i-th row 
                # of the M x M subgrid. This substring starts at column 'col_start' and has length 'm'.
                # In Python slicing, the end index is exclusive, so the slice is [col_start : col_start + m].
                s_sub_row_segment = grid_s[s_row_index][col_start : col_start + m]
                
                # Compare the extracted substring from S with the i-th row of T (grid_t[i]).
                if s_sub_row_segment != grid_t[i]:
                    # If the row segment in S does not match the corresponding row in T,
                    # then the subgrid starting at (row_start, col_start) does not match T.
                    # Mark it as not a match.
                    match = False
                    # There's no need to check the remaining rows for this potential top-left corner.
                    # Break out of the inner loop (the 'for i in range(m)' loop).
                    break 
            
            # After checking all M rows (or breaking early due to a mismatch):
            # If the 'match' flag is still True, it means all M rows of the subgrid matched 
            # the corresponding rows of T perfectly.
            if match:
                # We have found the unique matching subgrid, as guaranteed by the problem statement.
                # The problem asks for 1-based coordinates (a, b).
                # Convert the 0-based (row_start, col_start) to 1-based by adding 1.
                a = row_start + 1
                b = col_start + 1
                
                # Print the result in the required format "a b".
                print(f"{a} {b}")
                
                # Since exactly one match is guaranteed, we can stop the entire search process
                # immediately after finding the match.
                return

# Execute the solve function to read input, find the match, and print the output.
solve()

# END OF YOUR CODE HERE