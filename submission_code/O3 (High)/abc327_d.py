import sys
from collections import deque

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    it = iter(data)

    N = next(it)
    M = next(it)

    A = [next(it) for _ in range(M)]
    B = [next(it) for _ in range(M)]

    # Self-loop → impossible (X_v must differ from itself)
    for a, b in zip(A, B):
        if a == b:
            print("No")
            return

    # Build undirected graph
    adj = [[] for _ in range(N)]
    for a, b in zip(A, B):
        u = a - 1
        v = b - 1
        adj[u].append(v)
        adj[v].append(u)

    color = [-1] * N            # -1 : unvisited, 0/1 : two colors
    q = deque()

    for s in range(N):
        if color[s] != -1 or not adj[s]:
            continue            # already colored or isolated vertex
        color[s] = 0
        q.append(s)

        while q:
            v = q.popleft()
            for nxt in adj[v]:
                if color[nxt] == -1:
                    color[nxt] = color[v] ^ 1
                    q.append(nxt)
                elif color[nxt] == color[v]:
                    # same color on both ends → not bipartite
                    print("No")
                    return

    print("Yes")

if __name__ == "__main__":
    main()