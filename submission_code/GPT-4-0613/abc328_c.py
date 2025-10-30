import sys
from itertools import accumulate

N, Q = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()

cnt = [0] * (N + 1)
for i in range(1, N):
    if S[i - 1] == S[i]:
        cnt[i] = cnt[i - 1] + 1
    else:
        cnt[i] = cnt[i - 1]

for _ in range(Q):
    l, r = map(int, sys.stdin.readline().split())
    print(cnt[r - 1] - cnt[l - 1])