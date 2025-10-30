import sys

def solve():
    N = int(sys.stdin.readline())

    # Initialize the 3D array A to store input values.
    # A[x][y][z] (0-indexed) will correspond to A_{x+1,y+1,z+1} (1-indexed) from the problem description.
    A = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(N)]

    # Read the A values according to the specified input format.
    # The input iterates through x, then y, then z.
    for x in range(N):
        for y in range(N):
            # Each line contains N integers for a fixed (x,y) and varying z.
            line_values = list(map(int, sys.stdin.readline().split()))
            for z in range(N):
                A[x][y][z] = line_values[z]

    # Initialize the 3D prefix sum array S.
    # S[x][y][z] will store the sum of A_i,j,k for 1 <= i <= x, 1 <= j <= y, 1 <= k <= z.
    # We use 1-based indexing for S for easier application of the prefix sum formulas.
    # Its dimensions will be (N+1) x (N+1) x (N+1).
    S = [[[0 for _ in range(N + 1)] for _ in range(N + 1)] for _ in range(N + 1)]

    # Build the prefix sum array S using the inclusion-exclusion principle.
    for x_1based in range(1, N + 1):
        for y_1based in range(1, N + 1):
            for z_1based in range(1, N + 1):
                # Get the current A value, converting from 1-based S index to 0-based A index.
                current_A_val = A[x_1based - 1][y_1based - 1][z_1based - 1]

                # Apply the 3D prefix sum formula:
                # S(x,y,z) = A(x,y,z) + S(x-1,y,z) + S(x,y-1,z) + S(x,y,z-1)
                #            - S(x-1,y-1,z) - S(x-1,y,z-1) - S(x,y-1,z-1)
                #            + S(x-1,y-1,z-1)
                S[x_1based][y_1based][z_1based] = current_A_val \
                                                    + S[x_1based - 1][y_1based][z_1based] \
                                                    + S[x_1based][y_1based - 1][z_1based] \
                                                    + S[x_1based][y_1based][z_1based - 1] \
                                                    - S[x_1based - 1][y_1based - 1][z_1based] \
                                                    - S[x_1based - 1][y_1based][z_1based - 1] \
                                                    - S[x_1based][y_1based - 1][z_1based - 1] \
                                                    + S[x_1based - 1][y_1based - 1][z_1based - 1]

    Q = int(sys.stdin.readline())

    # Process each query
    results = []
    for _ in range(Q):
        # Read the query coordinates (Lx, Rx, Ly, Ry, Lz, Rz) which are 1-based.
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, sys.stdin.readline().split())

        # Calculate the sum for the given range using the 8-term inclusion-exclusion formula.
        # This formula efficiently retrieves the sum from the precomputed S array.
        current_sum = S[Rx][Ry][Rz] \
                    - S[Lx - 1][Ry][Rz] \
                    - S[Rx][Ly - 1][Rz] \
                    - S[Rx][Ry][Lz - 1] \
                    + S[Lx - 1][Ly - 1][Rz] \
                    + S[Lx - 1][Ry][Lz - 1] \
                    + S[Rx][Ly - 1][Lz - 1] \
                    - S[Lx - 1][Ly - 1][Lz - 1]
        results.append(current_sum)

    # Print all results, each on a new line.
    for res in results:
        sys.stdout.write(str(res) + "
")

# Call the solve function to execute the program.
solve()