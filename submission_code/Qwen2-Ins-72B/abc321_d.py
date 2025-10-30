import numpy as np

N, M, P = map(int, input().split())
A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

A = np.minimum(A, P - B)
B = np.sort(B)

ans = 0
for a in A:
    ind = np.searchsorted(B, a)
    ans += ind * a
    ans += np.sum(np.minimum(P, B[:M-ind]))
print(ans)