def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    idx = 0
    
    # Read N
    N = int(input_data[idx]); idx += 1
    
    # Read the 3D array A (1-based indexing)
    A = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                A[x][y][z] = int(input_data[idx])
                idx += 1
    
    # Build 3D prefix sum array (prefix[x][y][z] = sum of A up to (x, y, z))
    prefix = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                prefix[x][y][z] = (prefix[x-1][y][z] + prefix[x][y-1][z] + prefix[x][y][z-1]
                                   - prefix[x-1][y-1][z] - prefix[x-1][y][z-1] - prefix[x][y-1][z-1]
                                   + prefix[x-1][y-1][z-1] + A[x][y][z])
    
    # Function to get prefix sum up to (x, y, z)
    def get_sum(x, y, z):
        if x < 0 or y < 0 or z < 0:
            return 0
        if x == 0 or y == 0 or z == 0:
            return 0
        return prefix[x][y][z]
    
    # Read Q
    Q = int(input_data[idx]); idx += 1
    
    # Process queries
    out = []
    for _ in range(Q):
        Lx = int(input_data[idx]); idx += 1
        Rx = int(input_data[idx]); idx += 1
        Ly = int(input_data[idx]); idx += 1
        Ry = int(input_data[idx]); idx += 1
        Lz = int(input_data[idx]); idx += 1
        Rz = int(input_data[idx]); idx += 1
        
        ans = (get_sum(Rx, Ry, Rz)
               - get_sum(Lx-1, Ry, Rz)
               - get_sum(Rx, Ly-1, Rz)
               - get_sum(Rx, Ry, Lz-1)
               + get_sum(Lx-1, Ly-1, Rz)
               + get_sum(Lx-1, Ry, Lz-1)
               + get_sum(Rx, Ly-1, Lz-1)
               - get_sum(Lx-1, Ly-1, Lz-1))
        out.append(str(ans))
    
    print("
".join(out))

if __name__ == '__main__':
    main()