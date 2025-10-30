def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # Build 3D grid indexed from 1 to N
    # We'll create a 3D list with dimensions (N+1) x (N+1) x (N+1), where index 0 is unused.
    A = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                A[x][y][z] = int(next(it))
    
    # Create 3D prefix sums array P where:
    # P[x][y][z] = sum_{i=1}^{x} sum_{j=1}^{y} sum_{k=1}^{z} A[i][j][k]
    P = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            # Get cumulative sum for each z in current (x,y) cell
            for z in range(1, N+1):
                P[x][y][z] = (A[x][y][z]
                    + P[x-1][y][z] 
                    + P[x][y-1][z] 
                    + P[x][y][z-1]
                    - P[x-1][y-1][z] 
                    - P[x-1][y][z-1] 
                    - P[x][y-1][z-1]
                    + P[x-1][y-1][z-1])
    
    Q = int(next(it))
    
    out_lines = []
    for _ in range(Q):
        Lx = int(next(it))
        Rx = int(next(it))
        Ly = int(next(it))
        Ry = int(next(it))
        Lz = int(next(it))
        Rz = int(next(it))
        
        # use inclusion/exclusion principle on prefix sums
        total = (P[Rx][Ry][Rz]
                 - P[Lx - 1][Ry][Rz]
                 - P[Rx][Ly - 1][Rz]
                 - P[Rx][Ry][Lz - 1]
                 + P[Lx - 1][Ly - 1][Rz]
                 + P[Lx - 1][Ry][Lz - 1]
                 + P[Rx][Ly - 1][Lz - 1]
                 - P[Lx - 1][Ly - 1][Lz - 1])
        
        out_lines.append(str(total))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()