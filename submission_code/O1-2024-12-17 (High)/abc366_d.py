def main():
    import sys
    data = sys.stdin.read().split()
    idx = 0
    
    # Read N
    N = int(data[idx]); idx += 1
    
    # Read the 3D array A
    A = [[[0]*N for _ in range(N)] for _ in range(N)]
    for x in range(N):
        for y in range(N):
            row = data[idx:idx+N]
            idx += N
            for z in range(N):
                A[x][y][z] = int(row[z])
    
    # Build prefix sums P with an extra 1 in each dimension
    # so that P[x][y][z] holds sum of A[1..x, 1..y, 1..z].
    # We'll store them in P[x][y][z] with x,y,z in [0..N].
    # P[x][y][z] = sum of A[1..x][1..y][1..z].
    P = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                P[x][y][z] = (
                    A[x-1][y-1][z-1]
                    + P[x-1][y][z]
                    + P[x][y-1][z]
                    + P[x][y][z-1]
                    - P[x-1][y-1][z]
                    - P[x-1][y][z-1]
                    - P[x][y-1][z-1]
                    + P[x-1][y-1][z-1]
                )
    
    # Read Q
    Q = int(data[idx]); idx += 1
    
    # Process queries
    out = []
    for _ in range(Q):
        Lx = int(data[idx]); idx += 1
        Rx = int(data[idx]); idx += 1
        Ly = int(data[idx]); idx += 1
        Ry = int(data[idx]); idx += 1
        Lz = int(data[idx]); idx += 1
        Rz = int(data[idx]); idx += 1
        
        # Inclusion-exclusion to get the sum in O(1)
        ans = (
            P[Rx][Ry][Rz]
            - P[Lx-1][Ry][Rz]
            - P[Rx][Ly-1][Rz]
            - P[Rx][Ry][Lz-1]
            + P[Lx-1][Ly-1][Rz]
            + P[Lx-1][Ry][Lz-1]
            + P[Rx][Ly-1][Lz-1]
            - P[Lx-1][Ly-1][Lz-1]
        )
        out.append(str(ans))
    
    # Output
    print('
'.join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()