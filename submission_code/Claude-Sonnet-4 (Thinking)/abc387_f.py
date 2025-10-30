def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Convert to 0-indexed
    A = [a - 1 for a in A]
    
    MOD = 998244353
    
    # Find strongly connected components using Tarjan's algorithm
    def tarjan_scc():
        index_counter = [0]
        stack = []
        lowlinks = [-1] * N
        index = [-1] * N
        on_stack = [False] * N
        sccs = []
        
        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True
            
            w = A[v]  # Only one outgoing edge per node
            if index[w] == -1:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], index[w])
            
            if lowlinks[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
        
        for v in range(N):
            if index[v] == -1:
                strongconnect(v)
        
        return sccs
    
    sccs = tarjan_scc()
    
    # Map nodes to SCC indices
    scc_id = [-1] * N
    for i, scc in enumerate(sccs):
        for node in scc:
            scc_id[node] = i
    
    # Build SCC DAG
    num_sccs = len(sccs)
    scc_adj = [set() for _ in range(num_sccs)]
    for i in range(N):
        j = A[i]
        if scc_id[i] != scc_id[j]:
            scc_adj[scc_id[i]].add(scc_id[j])
    
    scc_adj = [list(s) for s in scc_adj]
    
    # DP to count valid assignments
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(scc, min_val):
        # Number of ways to assign values to SCC scc and all SCCs it can reach
        # such that SCC scc has value >= min_val
        
        if min_val > M:
            return 0
        
        total = 0
        for val in range(min_val, M + 1):
            ways = 1
            for child in scc_adj[scc]:
                ways = (ways * dp(child, val)) % MOD
            total = (total + ways) % MOD
        
        return total
    
    # Find SCCs with no incoming edges in the SCC DAG
    has_incoming = [False] * num_sccs
    for i in range(num_sccs):
        for j in scc_adj[i]:
            has_incoming[j] = True
    
    result = 1
    for i in range(num_sccs):
        if not has_incoming[i]:
            result = (result * dp(i, 1)) % MOD
    
    return result

print(solve())