def solve():
    n, k = map(int, input().split())
    
    if k == 1:
        print("Yes")
        return
    
    # Read edges and build adjacency list
    graph = [[] for _ in range(n * k + 1)]
    for _ in range(n * k - 1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # For k = 2, check if tree has a perfect matching
    if k == 2:
        matched = [False] * (n * k + 1)
        
        def dfs_matching(u, parent):
            unmatched_children = []
            for v in graph[u]:
                if v != parent:
                    if dfs_matching(v, u):
                        unmatched_children.append(v)
            
            if not matched[u] and unmatched_children:
                # Match u with first unmatched child
                matched[u] = True
                matched[unmatched_children[0]] = True
                return False
            
            return not matched[u]
        
        dfs_matching(1, -1)
        
        if all(matched[i] for i in range(1, n * k + 1)):
            print("Yes")
        else:
            print("No")
        return
    
    # For k >= 3, use backtracking to find paths
    used = [False] * (n * k + 1)
    paths_found = 0
    
    def find_path(start, current, path_len):
        if path_len == k:
            return True
        
        for next_v in graph[current]:
            if not used[next_v]:
                used[next_v] = True
                if find_path(start, next_v, path_len + 1):
                    return True
                used[next_v] = False
        
        return False
    
    def backtrack():
        nonlocal paths_found
        
        if paths_found == n:
            return True
        
        # Find unused vertex with minimum available degree
        start = -1
        min_degree = float('inf')
        
        for i in range(1, n * k + 1):
            if not used[i]:
                degree = sum(1 for j in graph[i] if not used[j])
                if degree < min_degree:
                    min_degree = degree
                    start = i
        
        if start == -1:
            return False
        
        # Try to build path from this vertex
        used[start] = True
        if find_path(start, start, 1):
            paths_found += 1
            if backtrack():
                return True
            paths_found -= 1
        
        # Backtrack - need to unmark all vertices in the failed path
        # Use DFS to find and unmark the partial path
        def unmark_path(v, visited):
            visited.add(v)
            used[v] = False
            for u in graph[v]:
                if used[u] and u not in visited:
                    # This neighbor might be part of the partial path
                    component_start = True
                    for w in graph[u]:
                        if w != v and used[w]:
                            component_start = False
                            break
                    if component_start:
                        unmark_path(u, visited)
        
        unmark_path(start, set())
        
        return False
    
    if backtrack():
        print("Yes")
    else:
        print("No")

solve()