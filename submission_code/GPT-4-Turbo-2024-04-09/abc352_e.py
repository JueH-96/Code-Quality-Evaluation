import sys
import heapq

input = sys.stdin.read
def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    edges = []
    
    for _ in range(M):
        K = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        vertices = list(map(int, data[index:index+K]))
        index += K
        
        # Generate all pairs (u, v) with u < v
        for i in range(K):
            for j in range(i + 1, K):
                u = vertices[i]
                v = vertices[j]
                edges.append((C, u, v))
    
    # Sort edges by weight
    edges.sort()
    
    # Kruskal's algorithm to find MST
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
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
    
    mst_weight = 0
    edges_used = 0
    
    for cost, u, v in edges:
        if union(u, v):
            mst_weight += cost
            edges_used += 1
            if edges_used == N - 1:
                break
    
    # Check if we used exactly N-1 edges (if the graph is connected)
    if edges_used == N - 1:
        print(mst_weight)
    else:
        print(-1)

if __name__ == "__main__":
    main()