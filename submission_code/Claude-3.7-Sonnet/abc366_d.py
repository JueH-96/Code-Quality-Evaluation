# Read input
N = int(input())

# Read the 3D array A
A = [[[0 for _ in range(N + 1)] for _ in range(N + 1)] for _ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        row = list(map(int, input().split()))
        for z in range(1, N + 1):
            A[x][y][z] = row[z - 1]

# Compute the 3D prefix sum array P
P = [[[0 for _ in range(N + 1)] for _ in range(N + 1)] for _ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            P[x][y][z] = (
                A[x][y][z] +
                P[x - 1][y][z] +
                P[x][y - 1][z] +
                P[x][y][z - 1] -
                P[x - 1][y - 1][z] -
                P[x - 1][y][z - 1] -
                P[x][y - 1][z - 1] +
                P[x - 1][y - 1][z - 1]
            )

# Process queries
Q = int(input())

for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
    
    result = (
        P[Rx][Ry][Rz] -
        P[Lx - 1][Ry][Rz] -
        P[Rx][Ly - 1][Rz] -
        P[Rx][Ry][Lz - 1] +
        P[Lx - 1][Ly - 1][Rz] +
        P[Lx - 1][Ry][Lz - 1] +
        P[Rx][Ly - 1][Lz - 1] -
        P[Lx - 1][Ly - 1][Lz - 1]
    )
    
    print(result)