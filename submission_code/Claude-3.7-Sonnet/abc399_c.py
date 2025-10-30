def main():
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    # Union-Find data structure
    parent = list(range(N + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    edges_to_remove = 0
    
    for u, v in edges:
        if find(u) == find(v):
            # Adding this edge would create a cycle
            edges_to_remove += 1
        else:
            union(u, v)
    
    print(edges_to_remove)

if __name__ == "__main__":
    main()