import sys

N, Q = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()
queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

for l, r in queries:
    count = 0
    for i in range(l, r):
        if S[i] == S[i+1]:
            count += 1
        else:
            count = 0
    print(count)