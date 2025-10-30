def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]
    
    # Adjacency list: for each node store (neighbor, length)
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1]); w = int(edges[idx+2])
        idx += 3
        adj[u].append((v,w))
        adj[v].append((u,w))
    
    # We will root the tree at node 1 and perform a BFS/DFS to:
    # 1) Find dist[v] = distance from 1 to v
    # 2) Find par[v] = parent of v in the BFS tree
    # 3) Store len2par[v] = length of the edge (v->par[v])
    
    from collections import deque
    dist = [0]*(N+1)
    par = [0]*(N+1)
    len2par = [0]*(N+1)
    visited = [False]*(N+1)
    
    visited[1] = True
    q = deque([1])
    while q:
        nd = q.popleft()
        for (nx, w) in adj[nd]:
            if not visited[nx]:
                visited[nx] = True
                par[nx] = nd
                dist[nx] = dist[nd] + w
                len2par[nx] = w
                q.append(nx)
    
    # We do not need par[1], set it to 0 or itself
    par[1] = 0
    len2par[1] = 0
    
    # We will sort all nodes (except 1) in descending order by dist[v].
    nodes_desc = sorted(range(2, N+1), key=lambda x: dist[x], reverse=True)
    
    covered = [False]*(N+1)  # covered[v] = whether v is fully accounted for in the union
    
    # We will create a list of "new coverage gains" G.
    # For each node in descending order, if it's not covered,
    # we climb up to the first covered ancestor (or until hitting node 1 if that is also not covered),
    # summing the edge lengths as we go, and mark each climbed node as covered.
    
    G = []
    
    for v in nodes_desc:
        if not covered[v]:
            # Climb up from v until we hit a covered node or we reach 1
            gain = 0
            cur = v
            while cur != 0 and (not covered[cur]):
                p = par[cur]
                gain += len2par[cur]
                covered[cur] = True
                cur = p
            if gain > 0:
                G.append(gain)
    
    # Sort coverage gains descending
    G.sort(reverse=True)
    # Build prefix sums of G
    prefix = [0]*(len(G)+1)
    for i in range(len(G)):
        prefix[i+1] = prefix[i] + G[i]
    
    # For each K from 1..N:
    # The maximal coverage is sum of the top min(K, len(G)) gains, because once we've
    # used up all gains, picking more vertices adds 0 additional coverage.
    
    # Then final answer = 2 * coverage.
    # We will output one line per K.
    
    out = []
    M = len(G)
    for K in range(1, N+1):
        take = K if K <= M else M  # number of nonzero-gain picks
        cover_val = prefix[take]
        score = 2 * cover_val
        out.append(str(score))
    
    print("
".join(out))

# Do not forget to call main().
if __name__ == "__main__":
    main()