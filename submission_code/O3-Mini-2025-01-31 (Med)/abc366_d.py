def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    # Read the integer N
    it = iter(data)
    N = int(next(it))
    
    # Create a 3D prefix sum array p with dimensions (N+1) x (N+1) x (N+1).
    # We use 1-indexing for simplicity.
    p = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(N + 1)]
    
    # Read the A values into p (temporarily hold the A values),
    # then we will build the prefix sum over them.
    # The input is given in order: for x=1, for each y=1..N, for each z=1..N then move to next x.
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                p[x][y][z] = int(next(it))
    
    # Build the 3D prefix sum.
    # p[x][y][z] will store the sum of all A[i][j][k] for 1 <= i <= x, 1 <= j <= y, 1 <= k <= z.
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                p[x][y][z] += (p[x - 1][y][z] + p[x][y - 1][z] + p[x][y][z - 1]
                               - p[x - 1][y - 1][z] - p[x - 1][y][z - 1] - p[x][y - 1][z - 1]
                               + p[x - 1][y - 1][z - 1])
    
    # Process the queries.
    Q = int(next(it))
    results = []
    for _ in range(Q):
        # Read the query tuple (Lx, Rx, Ly, Ry, Lz, Rz)
        Lx = int(next(it))
        Rx = int(next(it))
        Ly = int(next(it))
        Ry = int(next(it))
        Lz = int(next(it))
        Rz = int(next(it))
        
        # Using the 3D prefix sum, compute the sum in the sub-cuboid using inclusion-exclusion.
        sub_sum = (p[Rx][Ry][Rz]
                   - p[Lx - 1][Ry][Rz]
                   - p[Rx][Ly - 1][Rz]
                   - p[Rx][Ry][Lz - 1]
                   + p[Lx - 1][Ly - 1][Rz]
                   + p[Lx - 1][Ry][Lz - 1]
                   + p[Rx][Ly - 1][Lz - 1]
                   - p[Lx - 1][Ly - 1][Lz - 1])
        results.append(str(sub_sum))
    
    # Output each result in a new line.
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()