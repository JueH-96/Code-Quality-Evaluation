import numpy as np

N = int(input())
A = np.zeros((N+1, N+1, N+1), dtype=np.int64)
for i in range(1, N+1):
    for j in range(1, N+1):
        A[i, j, 1:N+1] = np.array(list(map(int, input().split())))
A = A.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

Q = int(input())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
    ans = A[Rx, Ry, Rz] - A[Rx, Ry, Lz-1] - A[Rx, Ly-1, Rz] - A[Rx, Ly-1, Lz-1] + A[Lx-1, Ry, Rz] + A[Lx-1, Ly-1, Rz] + A[Lx-1, Ry, Lz-1] - A[Lx-1, Ly-1, Lz-1]
    print(ans)