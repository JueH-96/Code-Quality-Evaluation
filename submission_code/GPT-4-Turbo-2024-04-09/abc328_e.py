import sys
import heapq

input = sys.stdin.read
def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    edges = []
    index = 3
    for _ in range(M):
        u = int(data[index]) - 1
        v = int(data[index+1]) - 1
        w = int(data[index+2])
        edges.append((w, u, v))
        index += 3
    
    # Kruskal's algorithm to find the Minimum Spanning Tree (MST)
    # Sort edges by weight
    edges.sort()
    
    # Union-Find data structure to manage connected components
    parent = list(range(N))
    rank = [0] * N
    
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
    
    mst_cost = 0
    edge_count = 0
    
    for w, u, v in edges:
        if union(u, v):
            mst_cost = (mst_cost + w) % K
            edge_count += 1
            if edge_count == N - 1:
                break
    
    print(mst_cost)

if __name__ == "__main__":
    main()