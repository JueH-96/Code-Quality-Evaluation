import sys

def solve():
    N = int(sys.stdin.readline())

    # P[x][y][z] will store the sum of A_ijk for 1<=i<=x, 1<=j<=y, 1<=k<=z.
    # Initialize (N+1)x(N+1)x(N+1) array with zeros.
    # Indices will be 1-based according to problem statement.
    # P[0][...][...], P[...][0][...], P[...][...][0] will remain 0, handling boundary cases.
    # Using x_dim, y_dim, z_dim as loop variables for clarity during P table construction.
    P = [[[0]*(N + 1) for _ in range(N + 1)] for _ in range(N + 1)]

    # Constructing the prefix sum table P
    # x_dim iterates for the first dimension (problem's x)
    # y_dim iterates for the second dimension (problem's y)
    # z_dim iterates for the third dimension (problem's z)
    for x_dim in range(1, N + 1): 
        for y_dim in range(1, N + 1): 
            # Read a line of N integers for A[x_dim][y_dim][1...N]
            line_values_str = sys.stdin.readline().split()
            for z_dim in range(1, N + 1): 
                A_xyz = int(line_values_str[z_dim-1]) # A_val[x_dim][y_dim][z_dim]
                
                # Compute P[x_dim][y_dim][z_dim] using the inclusion-exclusion principle
                P[x_dim][y_dim][z_dim] = A_xyz + \
                             P[x_dim-1][y_dim][z_dim] + \
                             P[x_dim][y_dim-1][z_dim] + \
                             P[x_dim][y_dim][z_dim-1] - \
                             P[x_dim-1][y_dim-1][z_dim] - \
                             P[x_dim-1][y_dim][z_dim-1] - \
                             P[x_dim][y_dim-1][z_dim-1] + \
                             P[x_dim-1][y_dim-1][z_dim-1]

    Q = int(sys.stdin.readline())
    results = [] # To store answers for fast printing at the end
    for _ in range(Q):
        coords_str = sys.stdin.readline().split()
        coords = [int(c) for c in coords_str] # map(int, ...) is also fine
        Lx, Rx, Ly, Ry, Lz, Rz = coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]

        # Querying the sum using the prefix sum table P
        # The ranges Lx, Rx refer to the first dimension (x_dim)
        # Ly, Ry refer to the second dimension (y_dim)
        # Lz, Rz refer to the third dimension (z_dim)
        current_sum = P[Rx][Ry][Rz] \
                    - P[Lx-1][Ry][Rz] \
                    - P[Rx][Ly-1][Rz] \
                    - P[Rx][Ry][Lz-1] \
                    + P[Lx-1][Ly-1][Rz] \
                    + P[Lx-1][Ry][Lz-1] \
                    + P[Rx][Ly-1][Lz-1] \
                    - P[Lx-1][Ly-1][Lz-1]
        results.append(str(current_sum))
    
    # Print all results, each on a new line
    sys.stdout.write("
".join(results) + "
")

solve()