def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    edges = data[1:]
    
    # Build adjacency list for the tree.
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
    
    # If N == 2, the answer is always 1, because either vertex 1 is a leaf or becomes one
    # trivially by removing the other leaf. But the general solution below covers this anyway.
    
    # We will root the tree at vertex 1, and compute "subtree_size[u]" for each u.
    # subtree_size[u] = number of vertices in the subtree rooted at u (including u).
    # Then, to remove vertex 1, we must remove all but at most one "child-subtree" of 1.
    # The cost of removing a whole child-subtree is simply its size, because each leaf-removal
    # operation removes exactly 1 vertex of that subtree until it's gone. 
    #
    # Finally, after removing all but one child-subtree of 1, vertex 1 itself becomes a leaf
    # (its degree = 1), and we remove it with one more operation. 
    #
    # Hence if children of 1 have subtree sizes s1, s2, ..., sk, then the minimal cost is
    # (s1 + s2 + ... + sk) - max(s1, s2, ..., sk) + 1.
    
    # Step 1: BFS (or DFS) from node 1 to establish parents in a tree structure.
    parent = [0] * (N+1)
    visited = [False] * (N+1)
    visited[1] = True
    
    queue = deque([1])
    bfs_order = []
    while queue:
        node = queue.popleft()
        bfs_order.append(node)
        for nxt in adj[node]:
            if not visited[nxt]:
                visited[nxt] = True
                parent[nxt] = node
                queue.append(nxt)
    
    # Step 2: Compute subtree sizes by processing nodes in reverse BFS order
    subtree_size = [0] * (N+1)
    while bfs_order:
        node = bfs_order.pop()
        subtree_size[node] = 1
        for nxt in adj[node]:
            if nxt != parent[node]:
                subtree_size[node] += subtree_size[nxt]
    
    # Collect subtree sizes of the immediate children of 1.
    costs = []
    for c in adj[1]:
        # If parent[c] == 1, c is indeed a direct child of 1 in our BFS tree.
        if parent[c] == 1:
            costs.append(subtree_size[c])
    
    # If 1 has no children (deg(1) == 0), it must be alone--but that cannot happen for N >= 2
    # Still handle the deg(1)=0 or deg(1)=1 scenario gracefully.
    if not costs:
        # deg(1) = 0  => 1 is alone (already contradictory if N>1),
        # but if it happens in edge case, we can remove 1 in one operation
        print(1)
        return
    
    # If 1 has children, the answer is sum of subtree sizes minus the largest subtree size plus 1
    total_sub = sum(costs)
    largest = max(costs)
    ans = total_sub - largest + 1
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()