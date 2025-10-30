import sys

# Increase the recursion limit to accommodate deep trees (N up to 2*10^5)
sys.setrecursionlimit(2 * 10**5 + 100) 

def solve():
    N, K = map(int, sys.stdin.readline().split())

    # Adjacency list to represent the tree
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Read the K specified target nodes
    target_nodes_raw = list(map(int, sys.stdin.readline().split()))
    
    # is_target[node] will be True if 'node' is one of the K specified vertices.
    # Using a boolean array for O(1) lookup.
    is_target = [False] * (N + 1)
    for node in target_nodes_raw:
        is_target[node] = True

    # has_target[u] will be True if the subtree rooted at 'u' (in the DFS tree
    # rooted at 1) contains any of the K specified target nodes.
    has_target = [False] * (N + 1)

    # DFS function to determine for each node 'u' if its subtree contains a target.
    # This DFS performs a post-order traversal to ensure all children's results
    # are available before computing for the parent.
    def dfs(u, parent):
        # A node 'u' itself is a target node, so its subtree contains a target.
        has_target[u] = is_target[u]
        
        # Iterate over neighbors (children in the DFS tree)
        for v in adj[u]:
            if v == parent:
                continue # Skip the parent to avoid going back up the tree
            
            # Recursively call DFS for child 'v'.
            # If the subtree rooted at 'v' contains a target (i.e., dfs(v, u) returns True),
            # then the subtree rooted at 'u' (its parent) also contains a target.
            if dfs(v, u):
                has_target[u] = True
        
        # Return the computed value for 'u'. This is used by its parent.
        return has_target[u]

    # Start DFS from an arbitrary root, e.g., vertex 1.
    # The parent of the root (1) is set to 0 (a non-existent node) to signify no parent.
    dfs(1, 0)

    # Count the number of nodes for which has_target is True.
    # These are precisely the nodes that form the minimal connected subgraph
    # (Steiner tree) containing all K specified vertices.
    ans = 0
    for i in range(1, N + 1):
        if has_target[i]:
            ans += 1
            
    # Print the result to standard output
    sys.stdout.write(str(ans) + "
")

# Call the solve function to execute the program
solve()