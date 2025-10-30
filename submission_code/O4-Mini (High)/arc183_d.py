import sys
import threading
def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a,b = map(int,input().split())
        adj[a].append(b)
        adj[b].append(a)

    # 1) BFS from 1 to get parent[], depth[], and children[]
    from collections import deque
    parent = [0]*(N+1)
    depth  = [0]*(N+1)
    children = [[] for _ in range(N+1)]
    q = deque([1])
    parent[1] = -1  # mark root
    depth[1] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if parent[v]==0:
                parent[v] = u
                depth[v] = depth[u] + 1
                children[u].append(v)
                q.append(v)

    # 2) Make a list of all nodes sorted by descending depth
    nodes = list(range(1,N+1))
    nodes.sort(key=lambda x: depth[x], reverse=True)

    # 3) We'll keep for each node u a max-heap (by depth) of
    #    currently-unmatched vertices in its subtree.
    #    In Python heapq is a *min*-heap, so we store (-depth[node], node).
    import heapq
    heap_of = [[] for _ in range(N+1)]

    # We'll collect our matched pairs here, plus the LCA-depth for sorting.
    pairs = []
    lca_d  = []

    # Process bottom-up:
    for u in nodes:
        Hu = heap_of[u]
        # Merge each child's heap into Hu, small->large
        for v in children[u]:
            Hv = heap_of[v]
            if len(Hu) < len(Hv):
                Hu, Hv = Hv, Hu
                heap_of[u], heap_of[v] = Hu, Hv
            # match all of Hv into Hu
            while Hv:
                d1,x = heapq.heappop(Hu)
                d2,y = heapq.heappop(Hv)
                pairs.append((x,y))
                lca_d.append(depth[u])
            # Hv is now empty; done with that child

        # Now push u itself as an unmatched vertex
        heapq.heappush(Hu, (-depth[u], u))

    # 4) Final round at the root (which is '1'),
    #    match the leftovers in heap_of[1] deepest<->shallowest
    rem = []
    Hroot = heap_of[1]
    while Hroot:
        _,node = heapq.heappop(Hroot)
        rem.append(node)
    # rem is in descending-depth order already
    k = len(rem)
    # k must be even (= 0 mod 2), we'll pair rem[i] with rem[k-1-i]
    for i in range(k//2):
        x = rem[i]
        y = rem[k-1-i]
        pairs.append((x,y))
        lca_d.append(depth[1])

    # Now we have exactly N/2 pairs.  Sort them by LCA-depth descending
    P = len(pairs)
    idx = list(range(P))
    idx.sort(key = lambda i: lca_d[i], reverse=True)

    # 5) Output
    out = []
    for i in idx:
        x,y = pairs[i]
        out.append(f"{x} {y}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()