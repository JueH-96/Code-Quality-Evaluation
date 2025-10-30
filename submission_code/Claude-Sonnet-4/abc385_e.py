from collections import defaultdict

def solve():
    n = int(input())
    
    # Build adjacency list
    adj = defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    min_deletions = float('inf')
    
    # Try each vertex as root
    for root in range(1, n + 1):
        neighbors = adj[root]
        
        # For each neighbor of root, count how many additional vertices it connects to
        intermediate_info = []
        for neighbor in neighbors:
            leaf_count = len(adj[neighbor]) - 1  # subtract 1 for connection to root
            intermediate_info.append(leaf_count)
        
        # Try different values of x and y
        for x in range(1, len(neighbors) + 1):
            # Sort to get the x intermediate vertices with most potential leaves
            intermediate_info_sorted = sorted(intermediate_info, reverse=True)
            
            if x > len(intermediate_info_sorted):
                continue
                
            # The maximum y is limited by the intermediate vertex with fewest leaves
            max_y = min(intermediate_info_sorted[:x]) if x > 0 else 0
            
            for y in range(0, max_y + 1):
                # Calculate kept vertices
                kept = 1 + x + x * y  # root + intermediate + leaves
                deletions = n - kept
                min_deletions = min(min_deletions, deletions)
    
    print(min_deletions)

solve()