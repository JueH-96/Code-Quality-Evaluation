# YOUR CODE HERE
import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    info = []
    for _ in range(m):
        info.append(list(map(int, sys.stdin.readline().split())))

    adj = [[] for _ in range(n + 1)]
    for a, b, x, y in info:
        adj[a].append((b, x, y))
        adj[b].append((a, -x, -y))

    coords = [(None, None)] * (n + 1)
    coords[1] = (0, 0)
    q = [1]
    visited = [False] * (n + 1)
    visited[1] = True

    while q:
        u = q.pop(0)
        for v, x, y in adj[u]:
            if coords[v] == (None, None):
                coords[v] = (coords[u][0] + x, coords[u][1] + y)
                if not visited[v]:
                    q.append(v)
                    visited[v] = True
            elif coords[v] != (coords[u][0] + x, coords[u][1] + y):
                print("undecidable")
                return

    for i in range(1, n + 1):
        if coords[i] == (None, None):
            print("undecidable")
        else:
            print(coords[i][0], coords[i][1])

solve()