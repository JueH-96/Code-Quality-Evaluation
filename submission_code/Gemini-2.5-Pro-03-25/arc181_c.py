import sys

# Function to read input and compute the grid
def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    
    # Read permutation P from standard input
    # P gives the order of rows based on lexicographical comparison of row strings.
    # P is a 1-based permutation, stored in a 0-indexed list. P[k] is the value P_{k+1}.
    P = list(map(int, sys.stdin.readline().split()))
    
    # Read permutation Q from standard input
    # Q gives the order of columns based on lexicographical comparison of column strings.
    # Q is a 1-based permutation, stored in a 0-indexed list. Q[k] is the value Q_{k+1}.
    Q = list(map(int, sys.stdin.readline().split()))

    # Compute inverse permutations to get ranks.
    # P_inv[i] will store the rank k such that P_k = i. This means row i is the k-th string in the sorted sequence S_{P_1} < ... < S_{P_N}.
    # Q_inv[j] will store the rank k such that Q_k = j. This means column j is the k-th string in the sorted sequence T_{Q_1} < ... < T_{Q_N}.
    # We use 1-based indexing for ranks and permutation values. Arrays are of size N+1.
    P_inv = [0] * (N + 1)
    Q_inv = [0] * (N + 1)

    # Populate P_inv: The k-th element specified in P (0-indexed k) is P[k].
    # This row P[k] has rank k+1 in the row string ordering. So P_inv[P[k]] = k+1.
    for k in range(N):
        P_inv[P[k]] = k + 1 
    
    # Populate Q_inv: The k-th element specified in Q (0-indexed k) is Q[k].
    # This column Q[k] has rank k+1 in the column string ordering. So Q_inv[Q[k]] = k+1.
    for k in range(N):
        Q_inv[Q[k]] = k + 1

    # Define row_rank and col_rank for clarity.
    # row_rank[i] is the rank of row i (1-based index i).
    row_rank = P_inv
    # col_rank[j] is the rank of column j (1-based index j).
    col_rank = Q_inv

    # Create the N x N grid A, initialized with 0s. Using 0-based indexing for the grid.
    # A[i][j] represents the character at row i+1, column j+1.
    A = [[0] * N for _ in range(N)]

    # Fill the grid based on the constructive rule proven to work:
    # A_{ij} = 1 if and only if P^{-1}_i + Q^{-1}_j > N.
    # Here, P^{-1}_i is the rank of row i, and Q^{-1}_j is the rank of column j.
    # In terms of 0-based grid indices: A[i][j] = 1 iff row_rank[i+1] + col_rank[j+1] > N.
    for i in range(N): # Corresponds to row i+1
        for j in range(N): # Corresponds to column j+1
            # Get the rank of row i+1
            r_i_plus_1 = row_rank[i + 1]
            # Get the rank of column j+1
            c_j_plus_1 = col_rank[j + 1]
            
            # Apply the condition using ranks
            if r_i_plus_1 + c_j_plus_1 > N:
                A[i][j] = 1
            # else: A[i][j] remains 0 (its initial value)

    # Print the resulting grid A to standard output.
    # Each row A[i] is converted to a string of '0's and '1's and printed on a new line.
    for i in range(N):
        # Use map to convert integers in row A[i] to strings
        # Use join to concatenate the strings without separators
        print("".join(map(str, A[i])))

# Execute the solve function to run the program
solve()