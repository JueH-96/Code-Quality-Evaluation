# YOUR CODE HERE
def main():
    N = int(input())
    
    # Read the 3D array
    A = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            row = list(map(int, input().split()))
            for z in range(1, N+1):
                A[x][y][z] = row[z-1]
    
    # Build 3D prefix sum array
    prefix = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                prefix[x][y][z] = A[x][y][z] + prefix[x-1][y][z] + prefix[x][y-1][z] + prefix[x][y][z-1] \
                                - prefix[x-1][y-1][z] - prefix[x-1][y][z-1] - prefix[x][y-1][z-1] \
                                + prefix[x-1][y-1][z-1]
    
    # Process queries
    Q = int(input())
    for _ in range(Q):
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
        
        # Calculate sum using inclusion-exclusion principle
        result = prefix[Rx][Ry][Rz]
        result -= prefix[Lx-1][Ry][Rz]
        result -= prefix[Rx][Ly-1][Rz]
        result -= prefix[Rx][Ry][Lz-1]
        result += prefix[Lx-1][Ly-1][Rz]
        result += prefix[Lx-1][Ry][Lz-1]
        result += prefix[Rx][Ly-1][Lz-1]
        result -= prefix[Lx-1][Ly-1][Lz-1]
        
        print(result)

if __name__ == "__main__":
    main()