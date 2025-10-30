import sys
from heapq import heappop, heappush

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = []
    for i in range(Q):
        L = int(data[2 + 3 * i]) - 1
        R = int(data[3 + 3 * i])
        C = int(data[4 + 3 * i])
        queries.append((L, R, C))
    
    # Union-Find to check connectivity
    parent = list(range(N + Q + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX
    
    # Kruskal's algorithm to find MST
    edges = []
    for i, (L, R, C) in enumerate(queries):
        for j in range(L, R):
            edges.append((C, N + i + 1, j + 1))
    
    edges.sort()
    
    mst_cost = 0
    for C, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_cost += C
    
    # Check if the graph is connected
    root = find(1)
    for i in range(1, N + Q + 1):
        if find(i) != root:
            print(-1)
            return
    
    print(mst_cost)

if __name__ == "__main__":
    main()