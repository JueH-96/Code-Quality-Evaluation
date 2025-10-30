import sys

N = int(input())
S = [input() for _ in range(N)]
M = max(len(s) for s in S)

T = [''] * M
for i in range(M):
    for j in range(N-1, -1, -1):
        if i < len(S[j]):
            T[i] += S[j][i]
        else:
            T[i] += '*'

for t in T:
    print(t)