import sys
from bisect import bisect_left

def read_input():
    N = int(sys.stdin.readline().strip())
    X = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())
    tasks = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]
    return N, X, Q, tasks

def solve():
    N, X, Q, tasks = read_input()
    X.sort()
    total_movements = 0

    for T, G in tasks:
        idx = bisect_left(X, G)
        if idx == T:
            continue
        elif idx == 0:
            total_movements += abs(G)
        elif idx == N:
            total_movements += abs(G - X[-1])
        else:
            total_movements += min(abs(G - X[idx-1]), abs(G - X[idx]))

    print(total_movements)

solve()