def solve():
    n = int(input())
    adj = [[] for _ in range(n+1)]
    
    for _ in range(n-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    max_snowflake_size = 0
    
    # Try each vertex as root
    for root in range(1, n+1):
        neighbors = adj[root]
        
        # Try each possible number of middle vertices
        from itertools import combinations
        for x in range(1, len(neighbors) + 1):
            # Try all combinations of x middle vertices
            for middle_vertices in combinations(neighbors, x):
                middle_set = set(middle_vertices)
                root_and_middle = {root} | middle_set
                
                # For each middle vertex, find valid leaves
                leaf_counts = []
                for middle in middle_vertices:
                    count = 0
                    for neighbor in adj[middle]:
                        if neighbor not in root_and_middle:
                            # Check if this neighbor is only connected to this middle vertex
                            # among root and middle vertices
                            connected_to_root_middle = set(adj[neighbor]) & root_and_middle
                            if connected_to_root_middle == {middle}:
                                count += 1
                    leaf_counts.append(count)
                
                # y is the minimum number of valid leaves among all middle vertices
                y = min(leaf_counts) if leaf_counts else 0
                
                size = 1 + x + x * y
                max_snowflake_size = max(max_snowflake_size, size)
    
    return n - max_snowflake_size

print(solve())