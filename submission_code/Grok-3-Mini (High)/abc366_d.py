import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
for x in range(1, N+1):
    for y in range(1, N+1):
        for z in range(1, N+1):
            A[x][y][z] = int(data[index])
            index += 1
prefix = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
for x in range(1, N+1):
    for y in range(1, N+1):
        for z in range(1, N+1):
            prefix[x][y][z] = (A[x][y][z] +
                               prefix[x-1][y][z] +
                               prefix[x][y-1][z] +
                               prefix[x][y][z-1] -
                               prefix[x-1][y-1][z] -
                               prefix[x-1][y][z-1] -
                               prefix[x][y-1][z-1] +
                               prefix[x-1][y-1][z-1])
Q = int(data[index])
index += 1
for _ in range(Q):
    Lx = int(data[index])
    index += 1
    Rx = int(data[index])
    index += 1
    Ly = int(data[index])
    index += 1
    Ry = int(data[index])
    index += 1
    Lz = int(data[index])
    index += 1
    Rz = int(data[index])
    index += 1
    sum_query = (prefix[Rx][Ry][Rz] -
                 prefix[Lx-1][Ry][Rz] -
                 prefix[Rx][Ly-1][Rz] -
                 prefix[Rx][Ry][Lz-1] +
                 prefix[Lx-1][Ly-1][Rz] +
                 prefix[Lx-1][Ry][Lz-1] +
                 prefix[Rx][Ly-1][Lz-1] -
                 prefix[Lx-1][Ly-1][Lz-1])
    print(sum_query)