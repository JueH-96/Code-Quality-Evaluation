# YOUR CODE HERE
import sys

# Read N and M from the first line of input.
# N is the size of the grid S (N x N).
# M is the size of the grid T (M x M).
N, M = map(int, sys.stdin.readline().split())

# Read grid S (N x N).
# S is stored as a list of strings, where each string represents a row.
# .strip() removes leading/trailing whitespace, including the newline character.
S = []
for _ in range(N):
    S.append(sys.stdin.readline().strip())

# Read grid T (M x M).
# T is stored similarly as a list of strings.
T = []
for _ in range(M):
    T.append(sys.stdin.readline().strip())

# Iterate through all possible top-left corners (r, c) of an M x M subgrid within S.
# Using 0-based indexing for the top-left corner (r, c):
# The row index r can range from 0 up to N - M.
# The column index c can range from 0 up to N - M.
# This is because if r > N - M, then r + M - 1 >= N - 1, meaning the subgrid would go beyond the bounds of S.
# The upper bound for range is exclusive, so we use N - M + 1.
for r in range(N - M + 1):
    for c in range(N - M + 1):
        # For the current potential top-left corner (r, c) in S,
        # check if the M x M subgrid starting at this position is identical to grid T.
        is_match = True
        
        # Iterate through the rows (i) and columns (j) of grid T (using 0-based indices).
        # These indices (i, j) represent the relative position within the M x M grid T.
        # The corresponding cell in grid S being compared is at (r + i, c + j).
        for i in range(M):
            for j in range(M):
                # Compare the character in S at (r + i, c + j) with the character in T at (i, j).
                if S[r + i][c + j] != T[i][j]:
                    # If a mismatch is found between any character in the subgrid and T,
                    # then this entire subgrid does not match T.
                    is_match = False
                    break # No need to check further cells within the current subgrid (inner loop over j)
            if not is_match:
                break # Mismatch found in the current row of the subgrid, break from the outer loop over i

        # If the inner loops completed without finding any mismatch (i.e., is_match is still True),
        # it means the M x M subgrid of S starting at (r, c) is identical to T.
        if is_match:
            # The problem guarantees that there is exactly one such subgrid.
            # The question asks for the 1-based indices (a, b) of the top-left corner.
            # If the 0-based indices are (r, c), the 1-based indices are (r + 1, c + 1).
            print(r + 1, c + 1)
            
            # Since we found the unique match, we can terminate the program immediately.
            sys.exit(0)

# The program is designed to find the unique match and exit using sys.exit(0).
# This part of the code should theoretically not be reached given the problem constraints and guarantee
# that exactly one solution exists.