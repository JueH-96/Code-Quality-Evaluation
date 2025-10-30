import numpy as np

N, M, P = map(int, input().split())
A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

A = np.sort(A)
B = np.sort(B)

posA = np.searchsorted(A, P)
sumA = np.append(A[:posA], P)
ans = np.sum(sumA)

if P < np.max(B):
    posB = np.searchsorted(B, P)
    resB = np.sum(np.append(B[posB:], P))
else:
    resB = np.sum(B)

idxB = range(M)
ans = ans * resB + posA * np.sum(B) - np.sum(np.minimum.outer(A[:posA], B[idxB]))

ans = np.sum(ans)

print(ans)