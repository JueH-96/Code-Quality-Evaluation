def min_deletions_for_snowflake_tree():
    n = int(input())
    
    # Construct the tree
    tree = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u, v = map(int, input().split())
        tree[u].append(v)
        tree[v].append(u)
    
    min_deletions = n
    
    # Try each vertex as the central vertex
    for central in range(1, n+1):
        # For each y (number of leaf children per intermediate vertex), compute the maximum x
        for y in range(n+1):  # y can range from 0 to n
            valid_intermediate_count = 0
            
            # Check how many potential intermediate vertices can have exactly y leaf children
            for neighbor in tree[central]:
                neighbor_count = 0
                for leaf_candidate in tree[neighbor]:
                    if leaf_candidate != central:
                        neighbor_count += 1
                
                if neighbor_count >= y:
                    valid_intermediate_count += 1
            
            if valid_intermediate_count > 0:
                # Compute the total number of vertices in the snowflake tree
                total_snowflake_vertices = 1 + valid_intermediate_count + valid_intermediate_count * y
                
                # Ensure we don't count more vertices than exist in the original tree
                if total_snowflake_vertices <= n:
                    # Compute the number of deleted vertices
                    deletions = n - total_snowflake_vertices
                    min_deletions = min(min_deletions, deletions)
    
    return min_deletions

if __name__ == "__main__":
    print(min_deletions_for_snowflake_tree())