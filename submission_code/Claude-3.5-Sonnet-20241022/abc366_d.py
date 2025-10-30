N = int(input())
A = [[[0]*N for _ in range(N)] for _ in range(N)]

# Read 3D array
for i in range(N):
    for j in range(N):
        row = list(map(int, input().split()))
        for k in range(N):
            A[i][j][k] = row[k]

Q = int(input())
for _ in range(Q):
    lx, rx, ly, ry, lz, rz = map(int, input().split())
    lx -= 1; rx -= 1; ly -= 1; ry -= 1; lz -= 1; rz -= 1
    
    # Calculate sum for current query
    total = 0
    for x in range(lx, rx+1):
        for y in range(ly, ry+1):
            for z in range(lz, rz+1):
                total += A[x][y][z]
    
    print(total)