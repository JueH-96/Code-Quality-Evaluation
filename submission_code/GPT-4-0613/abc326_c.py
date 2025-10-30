import sys
from collections import deque
from bisect import bisect_right

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
A.sort()

dq = deque()
max_gifts = 0
for i in range(N):
    while dq and dq[0] <= A[i] - M:
        dq.popleft()
    dq.append(A[i])
    max_gifts = max(max_gifts, len(dq))

print(max_gifts)