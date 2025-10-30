import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        edges.append((b / c, u, v))
    edges.sort(reverse=True)
    visited = [False] * (N + 1)
    visited[1] = True
    total_b = 0
    total_c = 0
    for b, u, v in edges:
        if not visited[v]:
            visited[v] = True
            total_b += b
            total_c += b * u + c
    print(total_b / total_c)

solve()