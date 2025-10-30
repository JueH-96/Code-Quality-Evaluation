n = int(input())

# Read the 3D array
A = []
for x in range(n):
    layer = []
    for y in range(n):
        row = list(map(int, input().split()))
        layer.append(row)
    A.append(layer)

# Build 3D prefix sum array
prefix = [[[0 for _ in range(n+1)] for _ in range(n+1)] for _ in range(n+1)]

for x in range(1, n+1):
    for y in range(1, n+1):
        for z in range(1, n+1):
            prefix[x][y][z] = (A[x-1][y-1][z-1] + 
                               prefix[x-1][y][z] + 
                               prefix[x][y-1][z] + 
                               prefix[x][y][z-1] - 
                               prefix[x-1][y-1][z] - 
                               prefix[x-1][y][z-1] - 
                               prefix[x][y-1][z-1] + 
                               prefix[x-1][y-1][z-1])

q = int(input())
for _ in range(q):
    lx, rx, ly, ry, lz, rz = map(int, input().split())
    
    result = (prefix[rx][ry][rz] - 
              prefix[lx-1][ry][rz] - 
              prefix[rx][ly-1][rz] - 
              prefix[rx][ry][lz-1] + 
              prefix[lx-1][ly-1][rz] + 
              prefix[lx-1][ry][lz-1] + 
              prefix[rx][ly-1][lz-1] - 
              prefix[lx-1][ly-1][lz-1])
    
    print(result)