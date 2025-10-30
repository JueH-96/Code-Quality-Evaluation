def solve():
    n = int(input())
    if n == 1:
        print(-1)
        return
    
    edges = []
    for _ in range(n - 1):
        a, b = map(int, input().split())
        edges.append((a, b))
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # Root the tree at vertex 1
    root = 1
    children = [[] for _ in range(n + 1)]
    visited = [False] * (n + 1)
    
    def dfs(v):
        visited[v] = True
        for u in adj[v]:
            if not visited[u]:
                children[v].append(u)
                dfs(u)
    
    dfs(root)
    
    # DP
    from functools import lru_cache
    
    @lru_cache(None)
    def dp(v, is_connected_to_parent, total_degree):
        # Returns (max_size, has_deg4)
        
        if not is_connected_to_parent and v != root:
            # v is not included in the alkane
            return (0, False)
        
        if total_degree not in [1, 4]:
            return (float('-inf'), False)
        
        if is_connected_to_parent:
            deg_to_children = total_degree - 1
        else:
            deg_to_children = total_degree
        
        if deg_to_children > len(children[v]) or deg_to_children < 0:
            return (float('-inf'), False)
        
        if len(children[v]) == 0:
            if deg_to_children == 0:
                return (1, total_degree == 4)
            else:
                return (float('-inf'), False)
        
        if deg_to_children == 0:
            return (1, total_degree == 4)
        
        # Compute the best value for each child
        child_values = []
        for child in children[v]:
            child_size_1, child_has_deg4_1 = dp(child, True, 1)
            child_size_4, child_has_deg4_4 = dp(child, True, 4)
            
            if child_size_1 > child_size_4:
                child_values.append((child_size_1, child_has_deg4_1))
            elif child_size_4 > child_size_1:
                child_values.append((child_size_4, child_has_deg4_4))
            elif child_size_1 == child_size_4 and child_size_1 > float('-inf'):
                child_values.append((child_size_4, child_has_deg4_4))
            else:
                child_values.append((float('-inf'), False))
        
        # Sort children by their values in descending order
        child_values.sort(reverse=True)
        
        if len(child_values) < deg_to_children or child_values[deg_to_children - 1][0] == float('-inf'):
            return (float('-inf'), False)
        
        total_size = 1
        total_has_deg4 = (total_degree == 4)
        for i in range(deg_to_children):
            total_size += child_values[i][0]
            total_has_deg4 = total_has_deg4 or child_values[i][1]
        
        return (total_size, total_has_deg4)
    
    # For the root, try all possible total degrees
    ans = float('-inf')
    for total_degree in [1, 4]:
        size, has_deg4 = dp(root, False, total_degree)
        if size > float('-inf') and has_deg4:
            ans = max(ans, size)
    
    if ans == float('-inf'):
        print(-1)
    else:
        print(ans)

solve()