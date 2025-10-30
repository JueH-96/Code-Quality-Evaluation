import sys

def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())

    # Read the N x N grid S
    S = []
    for _ in range(N):
        S.append(sys.stdin.readline().strip())

    # Read the M x M grid T
    T = []
    for _ in range(M):
        T.append(sys.stdin.readline().strip())

    # Iterate through all possible top-left corners (row_s, col_s) in grid S
    # where an M x M grid T could potentially start.
    # The valid 0-indexed row index (r_s) for the top-left corner of the subgrid
    # can range from 0 up to N - M (inclusive).
    # Similarly, the valid 0-indexed column index (c_s) can range from 0 up to N - M (inclusive).
    for r_s in range(N - M + 1):
        for c_s in range(N - M + 1):
            # Assume a match for the current subgrid position (r_s, c_s)
            is_match = True
            
            # Compare the M x M subgrid of S (starting at r_s, c_s) with grid T.
            # r_t and c_t are 0-indexed relative coordinates within T
            # (and also relative to the top-left corner of the subgrid in S).
            for r_t in range(M):
                for c_t in range(M):
                    # Check if the character at (r_s + r_t, c_s + c_t) in S
                    # is equal to the character at (r_t, c_t) in T.
                    if S[r_s + r_t][c_s + c_t] != T[r_t][c_t]:
                        is_match = False
                        # A mismatch is found, no need to check further columns in this row of the subgrid.
                        break 
                
                # If a mismatch was found in the inner loop (c_t loop),
                # then this entire (r_s, c_s) subgrid does not match T.
                # No need to check further rows (r_t loop) for the current (r_s, c_s).
                if not is_match:
                    break 
            
            # If after checking all cells in the M x M subgrid, is_match is still True,
            # it means we have found the unique match.
            if is_match:
                # Output the 1-indexed row and column as required by the problem.
                # Since r_s and c_s are 0-indexed, add 1 to convert to 1-indexed.
                print(r_s + 1, c_s + 1)
                # The problem guarantees exactly one pair, so we can exit after finding it.
                return 

# Call the solve function to execute the program logic
solve()