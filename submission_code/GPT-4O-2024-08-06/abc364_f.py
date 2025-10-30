# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    edges = []
    
    index = 2
    for i in range(1, Q + 1):
        L = int(data[index])
        R = int(data[index + 1])
        C = int(data[index + 2])
        index += 3
        
        for j in range(L, R + 1):
            edges.append((C, N + i, j))
    
    # Sort edges by cost
    edges.sort()
    
    # Union-Find (Disjoint Set Union) implementation
    parent = list(range(N + Q + 1))
    rank = [0] * (N + Q + 1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
            return True
        return False
    
    # Kruskal's algorithm to find MST
    mst_cost = 0
    edges_used = 0
    
    for cost, u, v in edges:
        if union(u, v):
            mst_cost += cost
            edges_used += 1
            if edges_used == N + Q - 1:
                break
    
    # Check if all N vertices are connected
    root_set = set(find(i) for i in range(1, N + 1))
    if len(root_set) == 1:
        print(mst_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()