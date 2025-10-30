import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = [[[int(x) for x in sys.stdin.readline().strip().split()] for _ in range(N)] for _ in range(N)]
    Q = int(sys.stdin.readline().strip())
    queries = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]

    prefix_sum = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                prefix_sum[x][y][z] = A[x-1][y-1][z-1] + prefix_sum[x-1][y][z] + prefix_sum[x][y-1][z] + prefix_sum[x][y][z-1] - prefix_sum[x-1][y-1][z] - prefix_sum[x-1][y][z-1] - prefix_sum[x][y-1][z-1] + prefix_sum[x-1][y-1][z-1]

    for Lx, Rx, Ly, Ry, Lz, Rz in queries:
        result = prefix_sum[Rx][Ry][Rz] - prefix_sum[Lx-1][Ry][Rz] - prefix_sum[Rx][Ly-1][Rz] - prefix_sum[Rx][Ry][Lz-1] + prefix_sum[Lx-1][Ly-1][Rz] + prefix_sum[Lx-1][Ry][Lz-1] + prefix_sum[Rx][Ly-1][Lz-1] - prefix_sum[Lx-1][Ly-1][Lz-1]
        print(result)

solve()