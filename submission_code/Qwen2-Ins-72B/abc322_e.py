from scipy.optimize import linear_sum_assignment
import numpy as np
N, K, P = map(int, input().split())
C = []
A = []
for _ in range(N):
    c, *a = map(int, input().split())
    C.append(c)
    A.append(a)
C = np.array(C)
A = np.array(A)
if K == 1:
    if np.any(A[:, 0] == P):
        print(np.min(C[A[:, 0] == P]))
    else:
        print(-1)
elif K == 2:
    if np.any((A[:, 0] + A[:, 1]) >= (P * 2)):
        row_ind, col_ind = linear_sum_assignment(C, True)
        cost = 0
        for i, j in zip(row_ind, col_ind):
            if (A[i, 0] + A[i, 1]) >= (P * 2):
                cost += C[i]
        print(cost)
    else:
        print(-1)
elif K == 3:
    if np.any((A[:, 0] + A[:, 1] + A[:, 2]) >= (P * 3)):
        row_ind, col_ind = linear_sum_assignment(C, True)
        cost = 0
        for i, j in zip(row_ind, col_ind):
            if (A[i, 0] + A[i, 1] + A[i, 2]) >= (P * 3):
                cost += C[i]
        print(cost)
    else:
        print(-1)
elif K == 4:
    if np.any((A[:, 0] + A[:, 1] + A[:, 2] + A[:, 3]) >= (P * 4)):
        row_ind, col_ind = linear_sum_assignment(C, True)
        cost = 0
        for i, j in zip(row_ind, col_ind):
            if (A[i, 0] + A[i, 1] + A[i, 2] + A[i, 3]) >= (P * 4):
                cost += C[i]
        print(cost)
    else:
        print(-1)
elif K == 5:
    if np.any((A[:, 0] + A[:, 1] + A[:, 2] + A[:, 3] + A[:, 4]) >= (P * 5)):
        row_ind, col_ind = linear_sum_assignment(C, True)
        cost = 0
        for i, j in zip(row_ind, col_ind):
            if (A[i, 0] + A[i, 1] + A[i, 2] + A[i, 3] + A[i, 4]) >= (P * 5):
                cost += C[i]
        print(cost)
    else:
        print(-1)