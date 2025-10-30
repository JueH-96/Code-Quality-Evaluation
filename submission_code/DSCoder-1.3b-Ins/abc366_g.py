import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Check if there is a way to write integers satisfying the condition
    for u in range(1, N+1):
        degree = [0]*(N+1)
        for v, w in edges:
            if u < v:
                degree[w] ^= 1
        if sum(degree) > 1:
            print("No")
            return
        else:
            print("Yes")
            print(degree.index(0) if sum(degree) == 1 else 1)

solve()