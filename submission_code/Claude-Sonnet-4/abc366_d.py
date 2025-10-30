# YOUR CODE HERE
N = int(input())

# Read the 3D array A
A = []
for x in range(N):
    layer = []
    for y in range(N):
        row = list(map(int, input().split()))
        layer.append(row)
    A.append(layer)

# Build 3D prefix sum array
# prefix[x][y][z] = sum of all A[i][j][k] where 0<=i<=x, 0<=j<=y, 0<=k<=z
prefix = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]

for x in range(1, N+1):
    for y in range(1, N+1):
        for z in range(1, N+1):
            prefix[x][y][z] = A[x-1][y-1][z-1] + \
                            prefix[x-1][y][z] + prefix[x][y-1][z] + prefix[x][y][z-1] - \
                            prefix[x-1][y-1][z] - prefix[x-1][y][z-1] - prefix[x][y-1][z-1] + \
                            prefix[x-1][y-1][z-1]

Q = int(input())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
    
    # Calculate sum using inclusion-exclusion principle
    result = prefix[Rx][Ry][Rz] - \
             prefix[Lx-1][Ry][Rz] - prefix[Rx][Ly-1][Rz] - prefix[Rx][Ry][Lz-1] + \
             prefix[Lx-1][Ly-1][Rz] + prefix[Lx-1][Ry][Lz-1] + prefix[Rx][Ly-1][Lz-1] - \
             prefix[Lx-1][Ly-1][Lz-1]
    
    print(result)