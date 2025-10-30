def solve(n, m, a):
    MOD = 998244353
    
    # Convert to 0-indexed
    a = [x - 1 for x in a]
    
    # Find SCCs using Tarjan's algorithm
    index_counter = [0]
    stack = []
    indices = [-1] * n
    lowlinks = [-1] * n
    on_stack = [False] * n
    sccs = []
    
    def strongconnect(v):
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        
        w = a[v]
        if indices[w] == -1:
            strongconnect(w)
            lowlinks[v] = min(lowlinks[v], lowlinks[w])
        elif on_stack[w]:
            lowlinks[v] = min(lowlinks[v], indices[w])
        
        if lowlinks[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)
    
    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)
    
    # Map each node to its SCC
    node_to_scc = [-1] * n
    for scc_id, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_id
    
    # Build the condensation graph
    scc_adj = [set() for _ in range(len(sccs))]
    for u in range(n):
        v = a[u]
        if node_to_scc[u] != node_to_scc[v]:
            scc_adj[node_to_scc[u]].add(node_to_scc[v])
    
    # Convert to lists
    scc_adj = [list(s) for s in scc_adj]
    
    # Find topological order
    in_degree = [0] * len(sccs)
    for u in range(len(sccs)):
        for v in scc_adj[u]:
            in_degree[v] += 1
    
    topo_order = []
    queue = []
    for i in range(len(sccs)):
        if in_degree[i] == 0:
            queue.append(i)
    
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in scc_adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # DP on the condensation graph
    dp = [[0] * (m + 1) for _ in range(len(sccs))]
    
    # Process in reverse topological order
    for scc_id in reversed(topo_order):
        for val in range(1, m + 1):
            if not scc_adj[scc_id]:  # No outgoing edges
                dp[scc_id][val] = 1
            else:
                ways = 1
                for child_scc in scc_adj[scc_id]:
                    child_ways = 0
                    for child_val in range(val, m + 1):
                        child_ways = (child_ways + dp[child_scc][child_val]) % MOD
                    ways = (ways * child_ways) % MOD
                dp[scc_id][val] = ways
    
    # Compute the answer
    # Special case: if all SCCs are singletons and there's a unique sink
    if len(sccs) == n:  # All nodes are in separate SCCs
        sinks = [i for i in range(len(sccs)) if not scc_adj[i]]
        if len(sinks) == 1:
            sink = sinks[0]
            # Check if all other SCCs point to the sink
            sources = [i for i in range(len(sccs)) if i != sink and sink in scc_adj[i]]
            if len(sources) == n - 1:
                # Special case: n-1 sources pointing to 1 sink
                answer = 0
                for sink_val in range(1, m + 1):
                    ways = pow(sink_val, n - 1, MOD)
                    answer = (answer + ways) % MOD
                return answer
    
    # General case
    # Find root SCCs
    is_root = [True] * len(sccs)
    for u in range(len(sccs)):
        for v in scc_adj[u]:
            is_root[v] = False
    
    answer = 1
    for scc_id in range(len(sccs)):
        if is_root[scc_id]:
            scc_ways = 0
            for val in range(1, m + 1):
                scc_ways = (scc_ways + dp[scc_id][val]) % MOD
            answer = (answer * scc_ways) % MOD
    
    return answer

# Read input
n, m = map(int, input().split())
a = list(map(int, input().split()))

# Solve and print the answer
print(solve(n, m, a))