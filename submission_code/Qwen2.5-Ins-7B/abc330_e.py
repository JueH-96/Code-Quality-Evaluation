# YOUR CODE HERE
from collections import Counter

N, Q = map(int, input().split())
A = list(map(int, input().split()))
cnt = Counter(A)
mex = 0

while cnt[mex]:
    mex += 1

for _ in range(Q):
    i, x = map(int, input().split())
    i -= 1
    if A[i] == mex:
        mex += 1
        while cnt[mex]:
            mex += 1
    A[i] = x
    cnt[A[i]] += 1
    cnt[x] -= 1
    while cnt[mex]:
        mex += 1
    print(mex)