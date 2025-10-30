import sys

def solve():
    """
    Reads grid data and finds the template grid T within the larger grid S.
    """
    # Read grid dimensions from standard input.
    try:
        N, M = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Gracefully handle potential empty lines or malformed input,
        # although the problem statement implies valid input.
        return

    # Read the N x N grid S as a list of strings.
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Read the M x M grid T (the template) as a list of strings.
    T = [sys.stdin.readline().strip() for _ in range(M)]

    # Iterate through all possible top-left corner positions (r, c) in S
    # where the template T could potentially fit.
    # The row index 'r' can range from 0 to N-M.
    # The column index 'c' can range from 0 to N-M.
    for r in range(N - M + 1):
        for c in range(N - M + 1):
            
            # For each position (r, c), assume the subgrid of S matches T
            # until a mismatch is found.
            is_match = True
            
            # Compare the M x M subgrid of S with T, row by row.
            for i in range(M):
                # Extract the i-th row of the subgrid from S.
                # This corresponds to the (r+i)-th row of S, from column c to c+M-1.
                s_subgrid_row = S[r + i][c : c + M]
                
                # If the subgrid row does not match the corresponding template row,
                # this position (r, c) is not the solution.
                if s_subgrid_row != T[i]:
                    is_match = False
                    break  # Exit the inner loop and try the next (r, c).
            
            # If the inner loop completed without finding any mismatch,
            # we have found the unique solution.
            if is_match:
                # The problem asks for 1-based indices (a, b).
                # Our loop variables (r, c) are 0-based, so we add 1.
                print(f"{r + 1} {c + 1}")
                return # Terminate after finding the solution.

# Execute the solution.
solve()