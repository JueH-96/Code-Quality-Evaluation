# YOUR CODE HERE
import sys

# Define the main logic within a function
def solve():
    # Read N from standard input, the size of the cubic grid dimension
    N = int(sys.stdin.readline())
    
    # Initialize the 3D array A using list comprehensions. Use N+1 size for 1-based indexing.
    # This array will store the input values A_{x,y,z}. Using N+1 allows us to use indices 1..N directly.
    # Initialize with 0s; P will rely on these boundary 0s implicitly.
    # We could potentially avoid storing A explicitly if memory is tight, 
    # by computing prefix sums directly while reading input, but for N=100 storing A is feasible.
    # A = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)] 
    # Instead of storing A, we can compute P directly to save some memory. Let's modify to do that.
    
    # Initialize the 3D prefix sum array P with zeros. Size (N+1)x(N+1)x(N+1).
    # P[x][y][z] will store the sum of A[i][j][k] for 1<=i<=x, 1<=j<=y, 1<=k<=z.
    P = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)]

    # Read the A values and compute prefix sums P simultaneously
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            # Read a line containing N space-separated integers for A[x][y][1]...A[x][y][N]
            line = list(map(int, sys.stdin.readline().split()))
            # Compute P[x][y][z] for z from 1 to N using the current A[x][y][z] value and previous P values.
            for z in range(1, N + 1):
                # Get the value A[x][y][z] from the input line. 
                # Input line is 0-indexed, so A[x][y][z] corresponds to line[z-1].
                current_A = line[z-1]
                
                # The recurrence relation applies the principle of inclusion-exclusion
                # to compute P[x][y][z] based on previously computed prefix sums and current_A.
                P[x][y][z] = current_A \
                             + P[x-1][y][z] + P[x][y-1][z] + P[x][y][z-1] \
                             - P[x-1][y-1][z] - P[x-1][y][z-1] - P[x][y-1][z-1] \
                             + P[x-1][y-1][z-1]

    # After this loop, the prefix sum array P is fully computed.
    # We didn't need to store the A array explicitly.

    # Read the number of queries Q
    Q = int(sys.stdin.readline())

    # Process each query
    results = [] # Store results to print them all at once at the end potentially faster
    for _ in range(Q):
        # Read the 6 integers defining the query cuboid: Lx, Rx, Ly, Ry, Lz, Rz
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, sys.stdin.readline().split())
        
        # Calculate the sum over the specified cuboid using the precomputed prefix sums P
        # This uses the 3D inclusion-exclusion formula on the prefix sums.
        # It involves 8 terms, corresponding to the 8 vertices derived from the corners 
        # (Lx-1, Ly-1, Lz-1) and (Rx, Ry, Rz) defining the query range sum based on origin (0,0,0) sums.
        sum_val = P[Rx][Ry][Rz] \
                  - P[Lx-1][Ry][Rz] \
                  - P[Rx][Ly-1][Rz] \
                  - P[Rx][Ry][Lz-1] \
                  + P[Lx-1][Ly-1][Rz] \
                  + P[Lx-1][Ry][Lz-1] \
                  + P[Rx][Ly-1][Lz-1] \
                  - P[Lx-1][Ly-1][Lz-1]
        
        # Append the calculated sum to the results list
        results.append(sum_val)

    # Print all collected results, each on a new line to standard output
    for res in results:
        print(res)

# Call the solve function to run the main logic when the script is executed
solve()