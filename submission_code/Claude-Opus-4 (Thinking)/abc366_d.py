# YOUR CODE HERE
N = int(input())

# Read the 3D array
A = [[[0] * (N+1) for _ in range(N+1)] for _ in range(N+1)]

for x in range(1, N+1):
    for y in range(1, N+1):
        line = list(map(int, input().split()))
        for z in range(1, N+1):
            A[x][y][z] = line[z-1]

# Build 3D prefix sum
prefix = [[[0] * (N+1) for _ in range(N+1)] for _ in range(N+1)]

for x in range(1, N+1):
    for y in range(1, N+1):
        for z in range(1, N+1):
            prefix[x][y][z] = (A[x][y][z] 
                             + prefix[x-1][y][z] 
                             + prefix[x][y-1][z] 
                             + prefix[x][y][z-1] 
                             - prefix[x-1][y-1][z] 
                             - prefix[x-1][y][z-1] 
                             - prefix[x][y-1][z-1] 
                             + prefix[x-1][y-1][z-1])

Q = int(input())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
    
    # Calculate sum using inclusion-exclusion
    result = (prefix[Rx][Ry][Rz]
            - prefix[Lx-1][Ry][Rz]
            - prefix[Rx][Ly-1][Rz]
            - prefix[Rx][Ry][Lz-1]
            + prefix[Lx-1][Ly-1][Rz]
            + prefix[Lx-1][Ry][Lz-1]
            + prefix[Rx][Ly-1][Lz-1]
            - prefix[Lx-1][Ly-1][Lz-1])
    
    print(result)