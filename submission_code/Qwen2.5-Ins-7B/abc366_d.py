# YOUR CODE HERE
import numpy as np

N = int(input())
A = np.zeros((N, N, N), dtype=int)
for i in range(N):
    for j in range(N):
        A[i, j, :] = list(map(int, input().split()))
for _ in range(N):
    for j in range(N):
        A[:, j, :] += A[:, j-1, :]
for i in range(N):
    A[i, :, :] += A[i-1, :, :]
for _ in range(N):
    A[:, i, :] += A[:, i-1, :]
for _ in range(N):
    A[:, :, i] += A[:, :, i-1]

Q = int(input())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
    print(A[Rx-1, Ry-1, Rz-1] - A[Lx-1, Ry-1, Rz-1] - A[Rx-1, Ly-1, Rz-1] + A[Lx-1, Ly-1, Rz-1])