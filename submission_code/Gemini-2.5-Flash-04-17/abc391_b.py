# YOUR CODE HERE
import sys

def solve():
    # Read N and M
    # N: size of grid S, M: size of grid T
    N, M = map(int, sys.stdin.readline().split())

    # Read grid S
    # S is an N x N grid represented as a list of N strings
    S = []
    for _ in range(N):
        S.append(sys.stdin.readline().strip())

    # Read grid T
    # T is an M x M grid represented as a list of M strings
    T = []
    for _ in range(M):
        T.append(sys.stdin.readline().strip())

    # Iterate through all possible top-left positions (a_prime, b_prime)
    # for the M x M subgrid within the N x N grid S.
    # a_prime is the 0-indexed row of the top-left cell of the subgrid.
    # b_prime is the 0-indexed column of the top-left cell of the subgrid.
    # The subgrid ranges from row a_prime to a_prime + M - 1
    # and column b_prime to b_prime + M - 1.
    # For the subgrid to be within S, we must have:
    # a_prime + M - 1 <= N - 1  =>  a_prime <= N - M
    # b_prime + M - 1 <= N - 1  =>  b_prime <= N - M
    # So, a_prime can range from 0 to N - M, and b_prime can range from 0 to N - M.
    for a_prime in range(N - M + 1):
        for b_prime in range(N - M + 1):
            
            # Assume the current subgrid starting at (a_prime, b_prime) matches T.
            match = True
            
            # Check if the M x M subgrid of S matches T cell by cell.
            # Iterate through rows (i_prime) of T (0-indexed from 0 to M-1)
            # Iterate through columns (j_prime) of T (0-indexed from 0 to M-1)
            for i_prime in range(M):
                for j_prime in range(M):
                    # Compare the character in S at the current subgrid position
                    # with the corresponding character in T.
                    # The cell in S corresponding to T[i_prime][j_prime] is at
                    # row index: a_prime + i_prime
                    # column index: b_prime + j_prime
                    if S[a_prime + i_prime][b_prime + j_prime] != T[i_prime][j_prime]:
                        # Mismatch found. This (a_prime, b_prime) is not the correct top-left corner.
                        match = False
                        # No need to check the rest of the cells for this (a_prime, b_prime).
                        # Break out of the inner loop (over j_prime).
                        break 
                
                # If mismatch was found in the inner loop (over j_prime),
                # also break out of the middle loop (over i_prime).
                if not match:
                    break 
            
            # If the inner loops completed without finding any mismatch, 'match' is still True.
            if match:
                # We found the correct top-left corner (a_prime, b_prime).
                # The problem asks for 1-indexed output (a, b).
                # a = a_prime + 1
                # b = b_prime + 1
                print(a_prime + 1, b_prime + 1)
                
                # The problem guarantees exactly one solution.
                # Once found, we can stop searching and terminate the program.
                return # Exit the function (and the script since this is the main call)

# Call the solve function to run the program logic
solve()