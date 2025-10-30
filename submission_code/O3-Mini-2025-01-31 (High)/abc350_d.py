def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    # Create DSU arrays for union-find
    parent = list(range(N+1))
    size = [1]*(N+1)
    
    def find(x):
        # Path compression iterative method
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    
    edges = []
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        union(a, b)
        edges.append((a, b))
    
    # Count the edges in each connected component.
    comp_edge_count = [0]*(N+1)
    for a, b in edges:
        rep = find(a)
        comp_edge_count[rep] += 1

    # In each connected component, we can eventually add edges until the subgraph becomes complete.
    # For a component with k vertices, the total number of edges in a clique is k*(k-1)//2.
    # Thus, the maximum number of new friendships (operations) that can be added is:
    #   clique_edges - current_edges
    # Sum these over all the connected components.
    ans = 0
    for i in range(1, N+1):
        if parent[i] == i:  # i is the representative for its component
            k = size[i]
            ans += (k*(k-1)//2) - comp_edge_count[i]
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()