# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read the 3D array A (0-indexed)
# The input is given block by block for x, then line by line for y, with N values per line for z
A = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(N)]
for x in range(N):
    for y in range(N):
        row = list(map(int, sys.stdin.readline().split()))
        for z in range(N):
            A[x][y][z] = row[z]

# Create the prefix sum array S (0-indexed, size (N+1)x(N+1)x(N+1))
# S[x][y][z] will store the sum of A[i][j][k] for 0 <= i < x, 0 <= j < y, 0 <= k < z
S = [[[0 for _ in range(N + 1)] for _ in range(N + 1)] for _ in range(N + 1)]

# Populate the prefix sum array S
# Iterate through the grid using 1-based logical coordinates (x, y, z) from 1 to N
# S[x][y][z] is calculated based on the element A[x-1][y-1][z-1] and previously computed S values
for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            # Inclusion-Exclusion principle to calculate S[x][y][z]
            # S[x][y][z] = sum over [0, x)x[0, y)x[0, z)
            # = element A[x-1][y-1][z-1] + sum of 3 adjacent smaller boxes
            # - sum of 3 adjacent smaller 2D faces (double counted)
            # + sum of 1 adjacent smaller 1D corner (triple counted and removed)
            S[x][y][z] = A[x-1][y-1][z-1] + \
                         S[x-1][y][z] + S[x][y-1][z] + S[x][y][z-1] - \
                         S[x-1][y-1][z] - S[x-1][y][z-1] - S[x][y-1][z-1] + \
                         S[x-1][y-1][z-1]

# Read Q
Q = int(sys.stdin.readline())

# Process queries
results = [] # Store results to print at the end
for _ in range(Q):
    # Read query bounds (1-based)
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, sys.stdin.readline().split())

    # Calculate sum using the prefix sum array
    # The query asks for the sum over A_{x,y,z} where Lx <= x <= Rx, Ly <= y <= Ry, Lz <= z <= Rz (1-based)
    # This corresponds to the sum over A[i][j][k] where Lx-1 <= i <= Rx-1, Ly-1 <= j <= Ry-1, Lz-1 <= k <= Rz-1 (0-based)
    # Using the prefix sum S[x][y][z] (sum over [0, x)x[0, y)x[0, z)),
    # the sum over the 0-based inclusive box [a, b]x[c, d]x[e, f] is
    # sum over [0, b+1)x[0, d+1)x[0, f+1)
    # - sum over [0, a)x[0, d+1)x[0, f+1)
    # - sum over [0, b+1)x[0, c)x[0, f+1)
    # - sum over [0, b+1)x[0, d+1)x[0, e)
    # + sum over [0, a)x[0, c)x[0, f+1)
    # + sum over [0, a)x[0, d+1)x[0, e)
    # + sum over [0, b+1)x[0, c)x[0, e)
    # - sum over [0, a)x[0, c)x[0, e)
    # Substitute a=Lx-1, b=Rx-1, c=Ly-1, d=Ry-1, e=Lz-1, f=Rz-1
    # This leads to indices a=Lx-1, b+1=Rx, c=Ly-1, d+1=Ry, e=Lz-1, f+1=Rz.
    # The formula is S[Rx][Ry][Rz] - S[Lx-1][Ry][Rz] - S[Rx][Ly-1][Rz] - S[Rx][Ry][Lz-1] + S[Lx-1][Ly-1][Rz] + S[Lx-1][Ry][Lz-1] + S[Rx][Ly-1][Lz-1] - S[Lx-1][Ly-1][Lz-1]
    total_sum = S[Rx][Ry][Rz] \
                - S[Lx-1][Ry][Rz] - S[Rx][Ly-1][Rz] - S[Rx][Ry][Lz-1] \
                + S[Lx-1][Ly-1][Rz] + S[Lx-1][Ry][Lz-1] + S[Rx][Ly-1][Lz-1] \
                - S[Lx-1][Ly-1][Lz-1]

    results.append(total_sum)

# Print results
for res in results:
    print(res)