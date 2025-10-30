import sys
import threading
from collections import deque

def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        # From A to B is (X, Y), so pos[B] = pos[A] + (X, Y)
        adj[A].append((B, X, Y))
        adj[B].append((A, -X, -Y))
    # To store whether we've fixed a node, and its coordinates
    visited = [False] * (N+1)
    posx = [0] * (N+1)
    posy = [0] * (N+1)
    # BFS from node 1
    dq = deque([1])
    visited[1] = True
    posx[1] = 0
    posy[1] = 0
    while dq:
        u = dq.popleft()
        ux, uy = posx[u], posy[u]
        for v, dx, dy in adj[u]:
            if not visited[v]:
                # pos[v] = pos[u] + (dx, dy)
                posx[v] = ux + dx
                posy[v] = uy + dy
                visited[v] = True
                dq.append(v)
    # Output
    out = []
    for i in range(1, N+1):
        if visited[i]:
            out.append(f"{posx[i]} {posy[i]}")
        else:
            out.append("undecidable")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()