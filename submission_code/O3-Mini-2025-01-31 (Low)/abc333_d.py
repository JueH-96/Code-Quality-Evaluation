def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build the undirected tree
    adj = [[] for _ in range(n+1)]
    for _ in range(n - 1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    
    # Special case: if vertex 1 is already a leaf then we can delete it in 1 operation.
    if len(adj[1]) <= 1:
        sys.stdout.write("1")
        return

    # In our process we are allowed to remove one leaf per operation.
    # Our goal is to remove vertex 1 as soon as possible.
    # To be able to remove vertex 1 (choosing it as a leaf) we must ensure that at most one of its neighbors is still present.
    # In other words, if vertex1 has d neighbors initially, we need to completely “clear out” at least (d-1) branches.
    #
    # Notice that if we delete every vertex except 1 and one of its branches then vertex1 becomes a leaf.
    # In the deletion process every vertex is removed exactly once.
    # Thus one valid deletion sequence is:
    #   - Delete every vertex except vertex1 and one branch (say the branch with maximum number of vertices).
    #   - Then delete vertex1.
    # This sequence takes (N - size(remaining branch)) + 0 (we need no extra op for the branch we keep) 
    #    and then one final operation to delete vertex1.
    # That is, answer = (N - size_max_branch) where size_max_branch is the maximum size among the subtrees attached to vertex1.
    #
    # We now compute for each neighbor v of 1 the size of the subtree with v (when 1 is not included).
    
    sys.setrecursionlimit(10**7)
    def dfs(u, parent):
        count = 1
        for nb in adj[u]:
            if nb == parent:
                continue
            count += dfs(nb, u)
        return count

    max_sub = 0
    for nb in adj[1]:
        # the DFS from neighbor nb will count every vertex in that branch (subtree)
        size_sub = dfs(nb, 1)
        if size_sub > max_sub:
            max_sub = size_sub
    # The deletion operations will be (total vertices except 1) minus the branch we keep,
    # plus the final deletion of vertex1.
    # That is, answer = ( (N - 1) - max_sub ) + 1 = N - max_sub.
    ans = n - max_sub
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()