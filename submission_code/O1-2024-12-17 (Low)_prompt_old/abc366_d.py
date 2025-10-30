def solve():
    import sys
    data = sys.stdin.read().strip().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    
    # Read the 3D array A (1-based indexing)
    A = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                A[x][y][z] = int(data[idx])
                idx += 1
    
    # Build 3D prefix sums array P
    P = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                P[x][y][z] = (P[x-1][y][z] + P[x][y-1][z] + P[x][y][z-1]
                              - P[x-1][y-1][z] - P[x-1][y][z-1] - P[x][y-1][z-1]
                              + P[x-1][y-1][z-1] + A[x][y][z])
    
    Q = int(data[idx])
    idx += 1
    
    # Process queries
    out = []
    for _ in range(Q):
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, data[idx:idx+6])
        idx += 6
        result = (  P[Rx][Ry][Rz]
                  - P[Lx-1][Ry][Rz]
                  - P[Rx][Ly-1][Rz]
                  - P[Rx][Ry][Lz-1]
                  + P[Lx-1][Ly-1][Rz]
                  + P[Lx-1][Ry][Lz-1]
                  + P[Rx][Ly-1][Lz-1]
                  - P[Lx-1][Ly-1][Lz-1] )
        out.append(str(result))
    
    print("
".join(out))

# Call solve()
solve()