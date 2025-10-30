import sys

N, Q = map(int, sys.stdin.readline().split())
P = [list(sys.stdin.readline().strip()) for _ in range(N)]
queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

for a, b, c, d in queries:
    count = 0
    for i in range(a, c+1):
        for j in range(b, d+1):
            if P[i%N][j%N] == 'B':
                count += 1
    print(count)