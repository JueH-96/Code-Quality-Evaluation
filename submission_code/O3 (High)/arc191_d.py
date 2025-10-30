import sys
from collections import deque

# ------------------------------------------------------------
def bfs(start, adj, n):
    """return array of distances from start (∞ -> -1)"""
    dist = [-1] * (n + 1)
    q = deque([start])
    dist[start] = 0
    while q:
        v = q.popleft()
        for w in adj[v]:
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist

# ------------------------------------------------------------
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    it = iter(sys.stdin.read().strip().split())
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    S = int(next(it))
    T = int(next(it))

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)

    # tiny graphs have to be handled explicitly
    if N == 2:
        # only possible graph is one edge S-T and S != T
        print(-1)               # cannot swap
        return

    if N == 3 and M == 2:
        # the only simple connected graph with N=3, M=2 is the path
        # S - x - T   (because S ≠ T and graph connected)
        # impossible to swap
        print(-1)
        return

    # general case
    distS = bfs(S, adj, N)
    dST = distS[T]
    # connected graph guarantees dST is defined

    ans = 2 * dST + 1
    print(ans)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()