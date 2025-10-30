N = int(input().strip())
A = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
for x in range(1, N+1):
    for y in range(1, N+1):
        a = list(map(int, input().strip().split()))
        for z in range(1, N+1):
            A[x][y][z] = a[z-1]

for x in range(1, N+1):
    for y in range(1, N+1):
        for z in range(1, N+1):
            A[x][y][z] += A[x-1][y][z] + A[x][y-1][z] + A[x][y][z-1] - A[x-1][y-1][z] - A[x-1][y][z-1] - A[x][y-1][z-1] + A[x-1][y-1][z-1]

Q = int(input().strip())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().strip().split())
    ans = A[Rx][Ry][Rz] - A[Lx-1][Ry][Rz] - A[Rx][Ly-1][Rz] - A[Rx][Ry][Rz-1] + A[Lx-1][Ly-1][Rz] + A[Lx-1][Ry][Rz-1] + A[Rx][Ly-1][Rz-1] - A[Lx-1][Ly-1][Rz-1]
    print(ans)