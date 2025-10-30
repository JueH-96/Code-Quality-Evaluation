import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, input().split())

    # reversed graph : for every edge a -> b we store b -> a
    rev_adj = [[] for _ in range(N + 1)]
    out_from_1 = []          # vertices that are direct successors of vertex 1

    for _ in range(M):
        a, b = map(int, input().split())
        rev_adj[b].append(a)          # reverse edge
        if a == 1:
            out_from_1.append(b)

    # Breadth-first search on the reversed graph starting from vertex 1
    dist = [-1] * (N + 1)
    dist[1] = 0
    dq = deque([1])

    while dq:
        v = dq.popleft()
        for nxt in rev_adj[v]:
            if dist[nxt] == -1:
                dist[nxt] = dist[v] + 1
                dq.append(nxt)

    # look for the best cycle that leaves vertex 1 with one edge (1 -> v)
    ans = float('inf')
    for v in out_from_1:
        if dist[v] != -1:                 # v can reach 1
            ans = min(ans, 1 + dist[v])   # edge (1 -> v) + shortest path v -> 1

    print(-1 if ans == float('inf') else ans)


if __name__ == "__main__":
    main()