from bisect import bisect_left
from itertools import accumulate

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

C = [0] * (10**9 + 1)
for i, a in enumerate(A):
    C[a] = i + 1
for i in range(10**9, 0, -1):
    C[i - 1] = C[i]
C[0] = N

B.sort()
A.sort()
A.append(10**9 + 1)
A = list(accumulate(A))

if C[B[-1]] < M:
    print(-1)
    exit()

ans = 0
for b in B:
    i = bisect_left(A, A[C[b] - 1] + b)
    if i == len(A) - 1:
        print(-1)
        exit()
    ans += A[i + 1] - A[i]
    C[A[i + 1]] -= 1

print(ans)