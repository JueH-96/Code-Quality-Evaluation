import sys

N, Q = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()
L = [0] * (N + 1)

for i in range(N - 1):
    L[i + 1] = L[i] + (1 if S[i] == S[i + 1] else 0)

for _ in range(Q):
    l, r = map(int, sys.stdin.readline().split())
    print(L[r - 1] - L[l - 1])