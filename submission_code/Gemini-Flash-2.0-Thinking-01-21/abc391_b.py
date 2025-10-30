# YOUR CODE HERE
import sys

# Read N and M
# N: size of grid S
# M: size of grid T
N, M = map(int, sys.stdin.readline().split())

# Read grid S (N lines, each with N characters)
# Store as a list of strings
S = []
for _ in range(N):
    S.append(sys.stdin.readline().strip())

# Read grid T (M lines, each with M characters)
# Store as a list of strings
T = []
for _ in range(M):
    T.append(sys.stdin.readline().strip())

# Iterate through all possible top-left corners (r, c) for the M x M subgrid within S.
# r and c are 0-indexed row and column respectively.
# The subgrid starting at S[r][c] occupies rows from r to r+M-1 and columns from c to c+M-1.
# These indices must be within the bounds of S (0 to N-1).
# r+M-1 <= N-1 => r <= N-M
# c+M-1 <= N-1 => c <= N-M
# So, r ranges from 0 to N-M, and c ranges from 0 to N-M.
for r in range(N - M + 1):
    for c in range(N - M + 1):
        # For the current top-left corner (r, c), check if the subgrid matches T.
        # Assume it matches initially.
        match_found_for_this_rc = True
        
        # Iterate through each cell (i, j) of the T grid (0-indexed).
        # i ranges from 0 to M-1 (row within T)
        # j ranges from 0 to M-1 (column within T)
        for i in range(M):
            for j in range(M):
                # Compare the cell T[i][j] with the corresponding cell in S.
                # The corresponding cell in S for T[i][j] when the top-left is (r, c) is S[r+i][c+j].
                # We access S using 0-indexed row r+i and column c+j.
                # We access T using 0-indexed row i and column j.
                if S[r + i][c + j] != T[i][j]:
                    # If any cell mismatches, this (r, c) is not the correct top-left corner.
                    match_found_for_this_rc = False
                    break # Mismatch found, no need to check further cells for this (r, c). Break the inner loop (over j).
            if not match_found_for_this_rc:
                break # Mismatch found in the inner loop, break the outer loop (over i) for this (r, c).

        # If the loops completed without finding any mismatch (match_found_for_this_rc is still True)
        if match_found_for_this_rc:
            # We found the unique pair (a, b) which is the 1-indexed equivalent of (r, c).
            # The problem requires 1-based output.
            # a = r + 1 (1-indexed row)
            # b = c + 1 (1-indexed column)
            print(r + 1, c + 1)
            
            # The problem guarantees exactly one solution. Once found, we can terminate the program.
            sys.exit()