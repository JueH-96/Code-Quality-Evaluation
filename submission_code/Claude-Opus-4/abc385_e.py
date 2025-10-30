def solve():
    n = int(input())
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    min_deletions = n  # worst case: delete all vertices
    
    # Try each vertex as the center
    for center in range(1, n + 1):
        # Get all neighbors of the center
        neighbors = adj[center]
        
        if len(neighbors) == 0:
            continue
        
        # For each neighbor, count how many leaves it has (excluding the center)
        leaf_counts = []
        for neighbor in neighbors:
            leaf_count = 0
            for next_vertex in adj[neighbor]:
                if next_vertex != center:
                    # Check if next_vertex is a leaf (degree 1)
                    if len(adj[next_vertex]) == 1:
                        leaf_count += 1
            leaf_counts.append(leaf_count)
        
        # Group neighbors by their leaf count
        from collections import Counter
        leaf_count_freq = Counter(leaf_counts)
        
        # Try each possible value of y (number of leaves per branch)
        for y, freq in leaf_count_freq.items():
            if y > 0:  # y must be positive
                # This means we have 'freq' branches with 'y' leaves each
                x = freq
                
                # Calculate vertices in this snowflake tree
                vertices_in_snowflake = 1 + x + x * y  # center + branches + leaves
                
                # Calculate deletions needed
                deletions = n - vertices_in_snowflake
                
                # Check if this forms a valid snowflake tree
                # We need to verify that the selected branches and leaves form the correct structure
                valid = True
                used_vertices = set([center])
                
                # Add the x branches that have exactly y leaves
                branch_count = 0
                for neighbor in neighbors:
                    leaf_count = 0
                    temp_leaves = []
                    for next_vertex in adj[neighbor]:
                        if next_vertex != center and len(adj[next_vertex]) == 1:
                            leaf_count += 1
                            temp_leaves.append(next_vertex)
                    
                    if leaf_count == y:
                        branch_count += 1
                        used_vertices.add(neighbor)
                        used_vertices.update(temp_leaves[:y])
                        if branch_count == x:
                            break
                
                if branch_count == x and len(used_vertices) == vertices_in_snowflake:
                    min_deletions = min(min_deletions, deletions)
    
    print(min_deletions)

solve()