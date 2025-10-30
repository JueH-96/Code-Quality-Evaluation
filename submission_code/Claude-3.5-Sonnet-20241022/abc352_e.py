class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = n
        self.components = n
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

def solve():
    N, M = map(int, input().split())
    edges = []
    
    # Process each operation
    for _ in range(M):
        K, C = map(int, input().split())
        vertices = list(map(lambda x: int(x)-1, input().split()))
        
        # Add edges between all pairs of vertices in the subset
        for i in range(K):
            for j in range(i+1, K):
                edges.append((C, vertices[i], vertices[j]))
    
    # Sort edges by weight for Kruskal's algorithm
    edges.sort()
    
    # Run Kruskal's algorithm
    uf = UnionFind(N)
    mst_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst_weight += weight
            
    # Check if graph is connected
    if uf.components > 1:
        print(-1)
    else:
        print(mst_weight)

solve()