def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # We'll parse N, M first
    N, M = map(int, input_data[:2])
    idx = 2
    
    # We will store (C_i, list_of_vertices)
    operations = []
    for _ in range(M):
        K_i = int(input_data[idx]); idx += 1
        C_i = int(input_data[idx]); idx += 1
        vertices = list(map(int, input_data[idx:idx+K_i]))
        idx += K_i
        operations.append((C_i, vertices))
    
    # Sort operations by their cost, ascending
    operations.sort(key=lambda x: x[0])
    
    # Disjoint Set Union (Union-Find) implementation
    parent = list(range(N+1))
    rank = [0]*(N+1)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    
    # Kruskal-like procedure
    edges_used = 0
    total_cost = 0
    for cost, group in operations:
        # We'll connect them in a chain: A_{1}, A_{2}, ..., A_{K_i}
        # This is enough to make them a single connected component in MST terms
        for i in range(1, len(group)):
            if union(group[i-1], group[i]):
                total_cost += cost
                edges_used += 1
                if edges_used == N - 1:
                    # MST complete
                    print(total_cost)
                    return
    
    # If we haven't used N-1 edges, the graph is not fully connected
    # Check if there's exactly one connected component
    # (Alternatively, we could just check edges_used)
    if edges_used < N - 1:
        print(-1)
    else:
        print(total_cost)

# Don't forget to call main()
if __name__ == "__main__":
    main()