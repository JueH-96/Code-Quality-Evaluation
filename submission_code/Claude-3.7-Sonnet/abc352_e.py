def find_minimum_spanning_tree_weight(n, operations):
    # Collect all edges with their minimum weights
    edges = {}  # (u, v) -> weight
    for ki, ci, vertices in operations:
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                edge = (u, v)
                if edge not in edges or ci < edges[edge]:
                    edges[edge] = ci
    
    # Union-find data structure for Kruskal's algorithm
    parent = list(range(n + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y
    
    # Sort edges by weight for Kruskal's algorithm
    sorted_edges = sorted([(weight, u, v) for (u, v), weight in edges.items()])
    
    # Apply Kruskal's algorithm to find MST
    mst_weight = 0
    edge_count = 0
    
    for weight, u, v in sorted_edges:
        if find(u) != find(v):
            union(u, v)
            mst_weight += weight
            edge_count += 1
            if edge_count == n - 1:
                # We've added all edges needed for MST
                return mst_weight
    
    # If we couldn't add n-1 edges, the graph isn't connected
    return -1

def main():
    n, m = map(int, input().split())
    operations = []
    
    for _ in range(m):
        ki, ci = map(int, input().split())
        vertices = list(map(int, input().split()))
        operations.append((ki, ci, vertices))
    
    result = find_minimum_spanning_tree_weight(n, operations)
    print(result)

if __name__ == "__main__":
    main()