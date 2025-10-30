import sys

# Read the size of the cube
N = int(input().strip())

# Read the cube values
A = [[[int(x) for x in input().strip().split()] for _ in range(N)] for _ in range(N)]

# Read the number of queries
Q = int(input().strip())

# Process each query
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().strip().split())
    # Adjust the indices to be zero-based
    Lx, Rx, Ly, Ry, Lz, Rz = Lx - 1, Rx - 1, Ly - 1, Ry - 1, Lz - 1, Rz - 1
    # Calculate the sum for the given range
    total_sum = sum(A[x][y][z] for x in range(Lx, Rx + 1) for y in range(Ly, Ry + 1) for z in range(Lz, Rz + 1))
    # Output the result
    print(total_sum)