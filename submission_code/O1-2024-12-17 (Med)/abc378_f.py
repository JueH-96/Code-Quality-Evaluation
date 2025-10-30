def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Edges (u_i, v_i), 1-based
    edges = input_data[1:]
    
    # Build adjacency (0-based internally)
    adj = [[] for _ in range(N)]
    deg_old = [0]*N
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]) - 1
        v = int(edges[idx+1]) - 1
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        deg_old[u] += 1
        deg_old[v] += 1
    
    # We only care about vertices whose original degree is 2 or 3
    # Build a mask "valid[v] = True" if deg_old[v] in {2,3}, else False
    valid = [ (deg_old[v] == 2 or deg_old[v] == 3) for v in range(N) ]
    
    # Build the "G'" adjacency restricted to valid vertices
    # We'll store it in a separate list gprime[v] = list of neighbors (still 0-based)
    gprime = [[] for _ in range(N)]
    for v in range(N):
        if valid[v]:
            for w in adj[v]:
                if valid[w]:
                    gprime[v].append(w)
    
    # Compute deg'(v) = number of neighbors in gprime
    degp = [len(gprime[v]) for v in range(N)]
    
    # We will "prune" vertices that have deg_old=2 but degp != 1.
    # Those cannot serve as valid endpoints (or internal) of the desired cycle.
    # Use a queue-based approach to remove them iteratively.
    from collections import deque
    Q = deque()
    
    # Initialize the queue
    for v in range(N):
        if valid[v] and deg_old[v] == 2 and degp[v] != 1:
            Q.append(v)
    
    while Q:
        v = Q.popleft()
        if not valid[v]:
            continue
        # Remove v from the graph G''
        valid[v] = False
        # Decrement neighbors' deg'
        for w in gprime[v]:
            if valid[w]:
                degp[w] -= 1
                if deg_old[w] == 2 and degp[w] != 1:
                    Q.append(w)
        gprime[v].clear()
    
    # Now "valid[v] = True" means v is in G''.
    # Next, we need to find connected components of G''.
    visited = [False]*N
    
    from collections import deque
    
    ans = 0
    
    for start in range(N):
        if valid[start] and not visited[start]:
            # BFS or DFS to find this component
            comp = []
            queue = deque([start])
            visited[start] = True
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nxt in gprime[cur]:
                    if valid[nxt] and not visited[nxt]:
                        visited[nxt] = True
                        queue.append(nxt)
            
            # In this component, collect all deg_old=2 leaves (those with degp=1)
            leaves = []
            for v in comp:
                if deg_old[v] == 2 and degp[v] == 1:
                    leaves.append(v)
            
            # If there are L such leaves, then the number of pairs is C(L,2)
            L = len(leaves)
            if L < 2:
                continue
            total_pairs = L*(L-1)//2
            
            # However, we must exclude pairs (x,y) that are neighbors in the original tree
            # because adding an edge between them would create a multi-edge (not simple).
            # In G'', two leaves can only be adjacent to exactly one neighbor each.
            # If two leaves are each other's neighbor, that's an invalid pair.
            
            leaf_set = set(leaves)
            bad_pairs = 0
            for x in leaves:
                # the single neighbor in G'':
                # (We're sure it has exactly 1 neighbor in G'' if degp[x] = 1.)
                nb_list = gprime[x]
                if nb_list:  # should have exactly 1 neighbor
                    w = nb_list[0]
                    # Check if w is also a leaf
                    if w in leaf_set:
                        # Then x <-> w is an adjacent-leaf pair
                        # Count it in bad_pairs in a symmetric-safe way
                        if x < w:
                            bad_pairs += 1
            
            valid_pairs = total_pairs - bad_pairs
            ans += valid_pairs
    
    print(ans)