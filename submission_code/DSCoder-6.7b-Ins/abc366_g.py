import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = defaultdict(list)
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        graph[u].append(v)
        graph[v].append(u)

    xors = [0] * (N+1)
    for i in range(1, N+1):
        xors[i] = sum(xors[j] for j in graph[i])

    if any(x % 2 for x in xors[1:]):
        print("No")
        return

    print("Yes")
    print(" ".join(str(1 << (N-1)) for _ in range(N)))

solve()