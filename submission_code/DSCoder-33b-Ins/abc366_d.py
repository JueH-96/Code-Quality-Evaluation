# YOUR CODE HERE

import sys

def solve():
    N = int(input())
    A = [[list(map(int, input().split())) for _ in range(N)] for _ in range(N)]
    Q = int(input())
    queries = [list(map(int, input().split())) for _ in range(Q)]

    # Create a 3D prefix sum array
    prefix_sum = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                prefix_sum[x][y][z] = prefix_sum[x-1][y][z] + prefix_sum[x][y-1][z] + prefix_sum[x][y][z-1] - prefix_sum[x-1][y-1][z] - prefix_sum[x-1][y][z-1] - prefix_sum[x][y-1][z-1] + prefix_sum[x-1][y-1][z-1] + A[x-1][y-1][z-1]

    # Process queries
    for query in queries:
        Lx, Rx, Ly, Ry, Lz, Rz = query
        result = prefix_sum[Rx][Ry][Rz] - prefix_sum[Lx-1][Ry][Rz] - prefix_sum[Rx][Ly-1][Rz] - prefix_sum[Rx][Ry][Lz-1] + prefix_sum[Lx-1][Ly-1][Rz] + prefix_sum[Lx-1][Ry][Lz-1] + prefix_sum[Rx][Ly-1][Lz-1] - prefix_sum[Lx-1][Ly-1][Lz-1]
        print(result)

solve()