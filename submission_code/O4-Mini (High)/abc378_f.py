import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # compute degrees
    deg = [0] * (N+1)
    for i in range(1, N+1):
        deg[i] = len(adj[i])

    # mark deg-3 nodes
    is_deg3 = [False] * (N+1)
    for i in range(1, N+1):
        if deg[i] == 3:
            is_deg3[i] = True

    # build graph of deg-3 nodes only (edges between two deg-3 nodes)
    g3adj = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        if is_deg3[u]:
            for v in adj[u]:
                if is_deg3[v]:
                    g3adj[u].append(v)

    visited = [False] * (N+1)
    ans = 0

    # for each connected component of deg-3 nodes, count its deg-2 neighbors
    for u in range(1, N+1):
        if not is_deg3[u] or visited[u]:
            continue
        # DFS over deg-3 subgraph
        stack = [u]
        visited[u] = True
        leaves = 0
        while stack:
            x = stack.pop()
            # count deg-2 neighbors of x in the original tree
            for w in adj[x]:
                if deg[w] == 2:
                    leaves += 1
            # traverse to other deg-3 neighbors
            for w in g3adj[x]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
        # from this component, any two distinct deg-2 neighbors form a valid pair
        if leaves >= 2:
            ans += leaves * (leaves - 1) // 2

    print(ans)

if __name__ == "__main__":
    main()