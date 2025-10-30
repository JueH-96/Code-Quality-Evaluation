def solve():
    N = int(input())
    A = []
    for _ in range(N):
        A.append([int(c) for c in input()])

    # B will be the new grid. Initialize B as a deep copy of A.
    # Inner cells are already correct with this copy.
    B = [row[:] for row in A]

    # The problem constraints state 2 <= N <= 100.
    # If N=1, no change would happen, but it's not a test case.

    # Update top row (excluding top-left, which gets value from left column)
    # A[0][j-1] (value to the left) moves to B[0][j] for j from 1 to N-1
    # Example for N=4: A[0][0]->B[0][1], A[0][1]->B[0][2], A[0][2]->B[0][3]
    if N > 1: # This check is technically not needed due to N>=2 constraint, but good for clarity.
        for j in range(1, N):
            B[0][j] = A[0][j-1]
        
        # Update right column (excluding top-right, which gets value from top row)
        # A[i-1][N-1] (value above) moves to B[i][N-1] for i from 1 to N-1
        # Example for N=4: A[0][3]->B[1][3], A[1][3]->B[2][3], A[2][3]->B[3][3]
        for i in range(1, N):
            B[i][N-1] = A[i-1][N-1]

        # Update bottom row (excluding bottom-right, which gets value from right column)
        # A[N-1][j+1] (value to the right) moves to B[N-1][j] for j from N-2 down to 0
        # Example for N=4: A[3][3]->B[3][2], A[3][2]->B[3][1], A[3][1]->B[3][0]
        for j in range(N - 2, -1, -1): # loop j = N-2, N-3, ..., 0
            B[N-1][j] = A[N-1][j+1]

        # Update left column (excluding bottom-left, which gets value from bottom row)
        # A[i+1][0] (value below) moves to B[i][0] for i from N-2 down to 0.
        # This also handles the top-left corner B[0][0] taking value from A[1][0].
        # Example for N=4: A[3][0]->B[2][0], A[2][0]->B[1][0], A[1][0]->B[0][0]
        for i in range(N - 2, -1, -1): # loop i = N-2, N-3, ..., 0
            B[i][0] = A[i+1][0]
            
    # Print the resulting grid B
    for i in range(N):
        print("".join(map(str, B[i])))

if __name__ == '__main__':
    solve()