def main():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build the tree using adjacency lists.
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    
    # Read the weights for each vertex.
    C = [0] * (n+1)
    for i in range(1, n+1):
        C[i] = int(next(it))
        
    total_weight = sum(C)
    
    # sub_w[u] will contain the total weight in the subtree of u.
    # f_val[u] will be the function value for vertex u.
    sub_w = [0] * (n+1)
    f_val = [0] * (n+1)
    
    # First DFS: Compute subtree weights and f(1) for an arbitrary root (vertex 1).
    def dfs1(u, parent, depth):
        sub_w[u] = C[u]
        curr = C[u] * depth
        for v in adj[u]:
            if v == parent:
                continue
            curr += dfs1(v, u, depth + 1)
            sub_w[u] += sub_w[v]
        return curr

    # f_val[1] is computed from the DFS starting at node 1.
    f_val[1] = dfs1(1, 0, 0)
    
    ans = f_val[1]
    
    # Second DFS: Re-rooting technique.
    # When moving from parent u to child v:
    #   f(v) = f(u) + (total_weight - 2 * sub_w[v])
    def dfs2(u, parent):
        nonlocal ans
        for v in adj[u]:
            if v == parent:
                continue
            f_val[v] = f_val[u] + (total_weight - 2 * sub_w[v])
            if f_val[v] < ans:
                ans = f_val[v]
            dfs2(v, u)
            
    dfs2(1, 0)
    
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()