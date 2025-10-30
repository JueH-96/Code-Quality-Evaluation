class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.size[px] < self.size[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        self.count -= 1
        return True

def solve():
    N, Q = map(int, input().split())
    edges = []
    
    # Read Q operations
    for i in range(Q):
        L, R, C = map(int, input().split())
        # For each vertex j in range [L,R], add edge between N+i+1 and j
        for j in range(L, R+1):
            edges.append((C, N+i+1, j))
    
    # Sort edges by cost
    edges.sort()
    
    # Kruskal's algorithm
    uf = UnionFind(N + Q + 1)
    total_cost = 0
    edges_used = 0
    
    for cost, u, v in edges:
        if uf.union(u, v):
            total_cost += cost
            edges_used += 1
    
    # Check if graph is connected
    # Need N+Q-1 edges for MST
    if uf.count > 1:
        print(-1)
    else:
        print(total_cost)

solve()