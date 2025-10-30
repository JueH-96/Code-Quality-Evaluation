import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    required = [0]*(N+1)
    Vs = [int(next(it)) for _ in range(K)]
    for v in Vs:
        required[v] = 1

    # Root the tree at any one required node (or 1 if none)
    root = Vs[0] if K>0 else 1

    parent = [0]*(N+1)
    order = []
    stack = [root]
    parent[root] = -1
    while stack:
        u = stack.pop()
        order.append(u)
        for w in adj[u]:
            if parent[w] == 0:
                parent[w] = u
                stack.append(w)

    # post-order accumulation
    cnt = [0]*(N+1)
    edges_used = 0
    # process in reverse order so children first
    for u in reversed(order):
        c = required[u]
        for w in adj[u]:
            if parent[w] == u:
                c += cnt[w]
        cnt[u] = c
        # for each child w, check if edge u-w is used
        if parent[u] > 0:
            # u is child, parent is parent[u]
            # cnt[u] is number of required in subtree u
            if cnt[u] > 0 and cnt[u] < K:
                edges_used += 1

    # number of vertices in the minimal subtree = edges + 1 (if K>0)
    if K == 0:
        print(0)
    else:
        print(edges_used + 1)

if __name__ == "__main__":
    main()