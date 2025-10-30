def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # We'll use a union-find (disjoint set union) to identify connected components.
    parent = list(range(n))
    size = [1] * n
    
    def find(x):
        # Path compression find.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        # Union by size.
        a = find(a)
        b = find(b)
        if a == b:
            return False
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        return True

    # Read all edges.
    edges = []
    for _ in range(m):
        u = int(next(it)) - 1  # switch to 0-indexed
        v = int(next(it)) - 1
        edges.append((u, v))
        union(u, v)
    
    # After processing edges, each connected component has a maximum possible number
    # of edges equal to: comp_size*(comp_size-1)//2 (complete graph).
    # In each component we already have some edges. However, note that the operation:
    # "Choose X, Y, Z such that X-Y and Y-Z exist but X-Z doesn't and add edge X-Z"
    # can only be applied if there is a common friend. It turns out that if you add edges
    # strategically, you can eventually convert any connected component into a complete graph.
    # Thus, for any connected component with s nodes and e existing edges,
    # the maximum extra friend connections possible is
    #   s*(s-1)//2 - e.
    # (This is true because even if some missing edge does not immediately have a common neighbor,
    # after adding some other missing edges it becomes possible.)
    
    # Count the component sizes and the number of edges in each component.
    comp_edge = {}
    comp_size = {}
    
    for i in range(n):
        root = find(i)
        comp_size[root] = comp_size.get(root, 0) + 1

    for u, v in edges:
        root = find(u)
        comp_edge[root] = comp_edge.get(root, 0) + 1

    # Calculate the answer by summing maximum extra edges for every connected component.
    ans = 0
    for comp in comp_size:
        s = comp_size[comp]
        e = comp_edge.get(comp, 0)
        ans += s * (s - 1) // 2 - e

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()