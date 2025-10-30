class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
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
    N, Q = map(int, input().split())
    
    edges = []
    
    # Generate all edges
    for i in range(Q):
        L, R, C = map(int, input().split())
        # Vertex N+i+1 connects to vertices L, L+1, ..., R
        # Convert to 0-indexed: vertex N+i connects to vertices L-1, L, ..., R-1
        for j in range(L-1, R):
            edges.append((C, N+i, j))
    
    # Sort edges by cost
    edges.sort()
    
    # Kruskal's algorithm
    uf = UnionFind(N + Q)
    mst_cost = 0
    edges_used = 0
    
    for cost, u, v in edges:
        if uf.union(u, v):
            mst_cost += cost
            edges_used += 1
            if edges_used == N + Q - 1:
                break
    
    # Check if graph is connected
    if uf.components == 1:
        print(mst_cost)
    else:
        print(-1)

solve()