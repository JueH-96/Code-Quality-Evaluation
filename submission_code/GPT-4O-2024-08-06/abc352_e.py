# YOUR CODE HERE
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    edges = []
    
    for _ in range(M):
        K_i = int(data[index])
        index += 1
        C_i = int(data[index])
        index += 1
        vertices = list(map(int, data[index:index + K_i]))
        index += K_i
        
        # Add all pairs (u, v) with weight C_i
        for i in range(K_i):
            for j in range(i + 1, K_i):
                u = vertices[i]
                v = vertices[j]
                edges.append((C_i, u, v))
    
    # Sort edges by weight
    edges.sort()
    
    # Initialize Union-Find
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    # Kruskal's algorithm to find MST
    mst_weight = 0
    edge_count = 0
    
    for weight, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_weight += weight
            edge_count += 1
            if edge_count == N - 1:
                break
    
    # Check if all nodes are connected
    root = find(parent, 1)
    for i in range(2, N + 1):
        if find(parent, i) != root:
            print(-1)
            return
    
    print(mst_weight)

main()