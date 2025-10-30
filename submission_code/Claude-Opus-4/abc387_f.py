def find_sccs(n, adj):
    # Kosaraju's algorithm
    visited = [False] * n
    stack = []
    
    def dfs1(v):
        visited[v] = True
        for u in adj[v]:
            if not visited[u]:
                dfs1(u)
        stack.append(v)
    
    for i in range(n):
        if not visited[i]:
            dfs1(i)
    
    # Create reverse graph
    rev_adj = [[] for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            rev_adj[u].append(v)
    
    visited = [False] * n
    sccs = []
    
    def dfs2(v, scc):
        visited[v] = True
        scc.append(v)
        for u in rev_adj[v]:
            if not visited[u]:
                dfs2(u, scc)
    
    while stack:
        v = stack.pop()
        if not visited[v]:
            scc = []
            dfs2(v, scc)
            sccs.append(scc)
    
    return sccs

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    # Convert to 0-indexed
    a = [x - 1 for x in a]
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        if a[i] != i:
            adj[i].append(a[i])
    
    # Find SCCs
    sccs = find_sccs(n, adj)
    
    # Map each node to its SCC index
    node_to_scc = [0] * n
    for scc_idx, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_idx
    
    # Build SCC graph
    scc_adj = [set() for _ in range(len(sccs))]
    for i in range(n):
        if a[i] != i:
            scc_i = node_to_scc[i]
            scc_j = node_to_scc[a[i]]
            if scc_i != scc_j:
                scc_adj[scc_i].add(scc_j)
    
    # Convert sets to lists
    scc_adj = [list(s) for s in scc_adj]
    
    # Topological sort of SCCs
    in_degree = [0] * len(sccs)
    for i in range(len(sccs)):
        for j in scc_adj[i]:
            in_degree[j] += 1
    
    queue = []
    for i in range(len(sccs)):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in scc_adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Process SCCs in reverse topological order
    MOD = 998244353
    scc_max_value = [m] * len(sccs)
    
    for scc_idx in reversed(topo_order):
        # Find minimum max value from dependencies
        for dep_scc in scc_adj[scc_idx]:
            scc_max_value[scc_idx] = min(scc_max_value[scc_idx], scc_max_value[dep_scc])
    
    # Calculate result
    result = 1
    for scc_idx in range(len(sccs)):
        result = (result * scc_max_value[scc_idx]) % MOD
    
    print(result)

solve()