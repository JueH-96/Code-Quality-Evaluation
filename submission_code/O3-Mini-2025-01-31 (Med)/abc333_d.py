def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Build an undirected tree.
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    
    # If vertex 1 is already a leaf, we can delete it immediately.
    if len(adj[1]) <= 1:
        sys.stdout.write("1")
        return

    # We now root the tree at vertex 1.
    # We want to compute for each neighbor v of 1 the size of its branch:
    # that is, the number of vertices in the subtree T(v) (when 1 is removed).
    subtree_size = [0]*(n+1)
    parent = [0]*(n+1)
    stack = [1]
    order = []
    parent[1] = -1  # mark the root's parent as -1
    while stack:
        node = stack.pop()
        order.append(node)
        for nei in adj[node]:
            if nei == parent[node]:
                continue
            parent[nei] = node
            stack.append(nei)
    # Now process nodes in postorder to compute subtree sizes.
    for node in reversed(order):
        s = 1  # count itself
        for nei in adj[node]:
            if nei == parent[node]:
                continue
            s += subtree_size[nei]
        subtree_size[node] = s

    # Among the direct children of 1, choose the branch with maximum size.
    best = 0
    for child in adj[1]:
        if parent[child] == 1:
            if subtree_size[child] > best:
                best = subtree_size[child]
                
    # Minimum operations = N - (size of saved branch)
    ans = n - best
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()