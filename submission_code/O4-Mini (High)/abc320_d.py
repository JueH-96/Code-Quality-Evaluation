import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b, X, Y = map(int, input().split())
        adj[a].append((b, X, Y))
        adj[b].append((a, -X, -Y))

    # coords[i] = (x_i, y_i) if reachable from 1; otherwise left at (0,0)
    x = [0] * (N+1)
    y = [0] * (N+1)
    reachable = [False] * (N+1)

    dq = deque([1])
    reachable[1] = True
    # Person 1 is at (0,0) by definition

    # BFS to assign coordinates to all nodes in component of 1
    while dq:
        u = dq.popleft()
        xu, yu = x[u], y[u]
        for v, dx, dy in adj[u]:
            if not reachable[v]:
                reachable[v] = True
                x[v] = xu + dx
                y[v] = yu + dy
                dq.append(v)

    out = []
    w = out.append
    for i in range(1, N+1):
        if reachable[i]:
            w(f"{x[i]} {y[i]}")
        else:
            w("undecidable")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()