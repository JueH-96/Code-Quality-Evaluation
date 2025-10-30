def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]

    # Build adjacency list and degree array
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)

    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        deg[u]+=1
        deg[v]+=1

    # We will form a new "forest" T' whose vertices are:
    #  - all nodes with original degree == 3
    #  - all nodes with original degree == 2 that have exactly ONE neighbor of degree == 3
    # In T', edges are only between:
    #  - degree-3 nodes that are adjacent
    #  - a degree-2 leaf (that qualifies) and its single degree-3 neighbor

    # First gather which deg=2 nodes qualify (i.e., have exactly one neighbor of deg=3).
    # We also mark deg=3 nodes as always included.
    is_in_Tprime = [False]*(N+1)

    # Count how many deg=3 neighbors each deg=2 node has
    count_deg3_neighbor = [0]*(N+1)
    for v in range(1, N+1):
        if deg[v] == 2:
            cnt = 0
            for w in adj[v]:
                if deg[w] == 3:
                    cnt += 1
            count_deg3_neighbor[v] = cnt

    # Mark deg=3 nodes
    for v in range(1, N+1):
        if deg[v] == 3:
            is_in_Tprime[v] = True

    # Mark deg=2 leaves (exactly one deg=3 neighbor)
    for v in range(1, N+1):
        if deg[v] == 2 and count_deg3_neighbor[v] == 1:
            is_in_Tprime[v] = True

    # Build adjacency list for T'
    Tprime = [[] for _ in range(N+1)]
    for v in range(1, N+1):
        if is_in_Tprime[v]:
            if deg[v] == 3:
                # Connect to all neighbors w that are also in T'
                for w in adj[v]:
                    if is_in_Tprime[w]:
                        # Must only connect if (v=3 & w=3) or (v=3 & w=2 with exactly 1 deg-3 neighbor)
                        # or (v=2 with 1 deg-3 neighbor & w=3). Because that's the only valid link.
                        Tprime[v].append(w)
            else:
                # v has deg=2 and is in T' => exactly one deg=3 neighbor
                # Only connect to that neighbor
                for w in adj[v]:
                    if deg[w] == 3 and is_in_Tprime[w]:
                        Tprime[v].append(w)
                        Tprime[w].append(v)

    # Find connected components of T'
    visited = [False]*(N+1)

    def dfs(start):
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nx in Tprime[node]:
                if not visited[nx]:
                    visited[nx] = True
                    stack.append(nx)
        return comp

    ans = 0
    for v in range(1, N+1):
        if is_in_Tprime[v] and not visited[v]:
            comp = dfs(v)
            # Count how many are deg=2 leaves in this component
            cnt_leaves = 0
            for x in comp:
                # In T': a node with original deg=2 is included only if it has exactly one deg=3 neighbor
                if deg[x] == 2 and count_deg3_neighbor[x] == 1:
                    cnt_leaves += 1
            # Number of ways to pick two distinct leaves is comb(cnt_leaves, 2)
            ans += cnt_leaves*(cnt_leaves-1)//2

    print(ans)

# Don't forget to call main()!
if __name__ == "__main__":
    main()