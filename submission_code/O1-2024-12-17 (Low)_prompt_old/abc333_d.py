def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])

    # Edge case: if N=2, it's straightforward to see the result directly
    # but our general logic will handle it anyway.
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 1
    for _ in range(N-1):
        u = int(input_data[idx]); v = int(input_data[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
    
    # If vertex 1 has degree <= 1, it is already a leaf => answer = 1
    if len(adj[1]) <= 1:
        print(1)
        return

    # Otherwise, we do one DFS from vertex 1 to compute subtree sizes.
    # subtree_size[u] = number of vertices in subtree rooted at u
    # when ignoring the edge towards its parent in the DFS tree.
    subtree_size = [0]*(N+1)
    visited = [False]*(N+1)

    def dfs(u, parent):
        visited[u] = True
        sz = 1
        for w in adj[u]:
            if w != parent:
                sz += dfs(w, u)
        subtree_size[u] = sz
        return sz

    # Run DFS from 1 (treating 1 as the root)
    dfs(1, -1)

    # Now for each neighbor of 1, subtree_size[neighbor]
    # is the size of that connected component if we "cut" the edge (1 -- neighbor).
    # We pick the maximum such size.
    max_subtree = 0
    for nei in adj[1]:
        max_subtree = max(max_subtree, subtree_size[nei])
    
    # The answer = N - max_subtree
    # Explanation:
    # We will keep the neighbor (and all its subtree) that yields max_subtree,
    # and remove all other neighbors' subtrees. Then 1 becomes a leaf (since it has exactly
    # one neighbor left), and we remove 1 in one final operation. So total = N - max_subtree.
    # (This is the minimal number of leaf-deletions needed to get rid of 1.)
    ans = N - max_subtree
    print(ans)

# For the online judge / testing environment, just call solve().
# (No extra top-level code is needed.)
if __name__ == "__main__":
    solve()