import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    weights = [(u, v, w) for u, v, w in edges]
    weights.sort(key=lambda x: x[2])
    parent = list(range(N+1))
    size = [1] * (N+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        x, y = find(x), find(y)
        if x == y:
            return False
        if size[x] < size[y]:
            x, y = y, x
        parent[y] = x
        size[x] += size[y]
        return True

    ans = 0
    for u, v, w in weights:
        if union(u, v):
            ans = (ans + w) % K
    print(ans)

solve()