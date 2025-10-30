def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    # Parser
    idx = 0
    N = int(input_data[idx]); idx+=1
    M = int(input_data[idx]); idx+=1
    
    subsets = []
    for _ in range(M):
        K_i = int(input_data[idx]); idx+=1
        C_i = int(input_data[idx]); idx+=1
        vertices = list(map(int, input_data[idx:idx+K_i]))
        idx += K_i
        subsets.append((C_i, vertices))
    
    # Sort by weight
    subsets.sort(key=lambda x: x[0])
    
    # Disjoint Set (Union-Find) structure
    parent = list(range(N+1))
    rank = [0]*(N+1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        rootA = find(a)
        rootB = find(b)
        if rootA != rootB:
            if rank[rootA] < rank[rootB]:
                parent[rootA] = rootB
            elif rank[rootA] > rank[rootB]:
                parent[rootB] = rootA
            else:
                parent[rootB] = rootA
                rank[rootA] += 1
            return True
        return False
    
    mst_weight = 0
    edge_count = 0
    
    # Kruskal-like approach but using the fact each subset forms a clique,
    # so we only need a chain of edges in ascending vertex order within each subset
    for C_i, vertices in subsets:
        vertices.sort()
        prev = vertices[0]
        for v in vertices[1:]:
            if union(prev, v):
                mst_weight += C_i
                edge_count += 1
                if edge_count == N - 1:
                    # Early stop if MST is complete
                    break
            prev = v
        if edge_count == N - 1:
            break
    
    # Check if the graph is connected: either edge_count == N-1 or
    # check if all vertices lie in the same connected component
    if edge_count < N - 1:
        print(-1)
    else:
        print(mst_weight)

# Call main() to adhere to the requirement
main()