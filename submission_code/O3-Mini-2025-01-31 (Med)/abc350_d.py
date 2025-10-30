def main():
    import sys

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # DSU initialization: each node is its own parent.
    parent = list(range(n))
    # Size array to track the number of nodes in each component.
    size = [1] * n

    def find(a):
        # Path compression find.
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        # Union by size.
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    edges = []
    for _ in range(m):
        # Converting to 0-based index.
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        edges.append((u, v))
        union(u, v)

    # Count the number of edges for each connected component.
    comp_edges = {}
    for u, v in edges:
        root = find(u)
        comp_edges[root] = comp_edges.get(root, 0) + 1

    # For each connected component, the final state would be a complete graph.
    # A complete graph on k nodes has k*(k-1)//2 edges.
    # So for a component with currently e edges, we can add 
    # k*(k-1)//2 - e new edges.
    answer = 0
    seen = set()
    for i in range(n):
        root = find(i)
        if root in seen:
            continue
        seen.add(root)
        k = size[root]
        current_edges = comp_edges.get(root, 0)
        answer += (k * (k - 1) // 2 - current_edges)
    
    sys.stdout.write(str(answer))


if __name__ == '__main__':
    main()