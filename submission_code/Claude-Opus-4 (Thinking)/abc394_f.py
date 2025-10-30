def solve():
    n = int(input())
    if n == 1:
        print(-1)
        return
    
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # Check if any vertex has degree >= 4
    if not any(len(adj[i]) >= 4 for i in range(1, n + 1)):
        print(-1)
        return
    
    max_size = -1
    
    # Try each vertex as root
    for root in range(1, n + 1):
        # Build tree structure from this root
        parent = [-1] * (n + 1)
        children = [[] for _ in range(n + 1)]
        visited = [False] * (n + 1)
        
        # DFS to build tree
        def build_tree(u, p):
            visited[u] = True
            parent[u] = p
            for v in adj[u]:
                if not visited[v]:
                    children[u].append(v)
                    build_tree(v, u)
        
        build_tree(root, -1)
        
        # DP: dp[u][parent_used][num_children_used] = max alkane size in subtree of u
        # parent_used: whether edge to parent is used (0 or 1)
        # num_children_used: number of edges to children used
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(u, parent_used):
            # Returns list of (num_children_used, max_size, has_degree4) tuples
            results = []
            
            num_children = len(children[u])
            
            # Try all combinations of children
            for mask in range(1 << num_children):
                children_used = []
                total_size = 0
                has_degree4 = False
                valid = True
                
                # Process children that are connected to u in alkane
                for i in range(num_children):
                    if mask & (1 << i):
                        children_used.append(i)
                
                # For each child, get best configuration
                child_sizes = []
                for i in range(num_children):
                    child = children[u][i]
                    if i in children_used:
                        # Child is connected to u
                        child_results = dp(child, 1)
                        best = -1
                        best_has_deg4 = False
                        for nc, sz, hd4 in child_results:
                            deg = 1 + nc  # degree of child in alkane
                            if deg in [1, 4] and sz > best:
                                best = sz
                                best_has_deg4 = hd4
                        if best == -1:
                            valid = False
                            break
                        child_sizes.append((best, best_has_deg4))
                    else:
                        # Child is not connected to u
                        child_results = dp(child, 0)
                        best = 0
                        best_has_deg4 = False
                        for nc, sz, hd4 in child_results:
                            if nc == 0 or nc == 1 or nc == 4:
                                if sz > best:
                                    best = sz
                                    best_has_deg4 = hd4
                        child_sizes.append((best, best_has_deg4))
                
                if not valid:
                    continue
                
                # Calculate total
                for sz, hd4 in child_sizes:
                    total_size += sz
                    has_degree4 |= hd4
                
                # Check u's degree
                u_degree = parent_used + len(children_used)
                
                if u_degree == 0:
                    # u not in alkane
                    results.append((0, total_size, has_degree4))
                elif u_degree == 1:
                    # u in alkane with degree 1
                    results.append((len(children_used), total_size + 1, has_degree4))
                elif u_degree == 4:
                    # u in alkane with degree 4
                    results.append((len(children_used), total_size + 1, True))
            
            return results
        
        # Get results for root
        root_results = dp(root, 0)
        for nc, sz, hd4 in root_results:
            if hd4 and (nc == 0 or nc == 1 or nc == 4):
                max_size = max(max_size, sz)
    
    print(max_size)

solve()