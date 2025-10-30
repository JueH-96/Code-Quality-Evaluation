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
    
    def is_connected(self):
        return self.components == 1

N, M = map(int, input().split())

edges = []

for _ in range(M):
    K, C = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Add all pairs (u, v) with weight C
    for i in range(K):
        for j in range(i + 1, K):
            u, v = A[i], A[j]
            edges.append((C, u - 1, v - 1))  # Convert to 0-indexed

# Sort edges by weight for Kruskal's algorithm
edges.sort()

# Use Union-Find to build MST and check connectivity
uf = UnionFind(N)
mst_weight = 0
edges_added = 0

for weight, u, v in edges:
    if uf.union(u, v):
        mst_weight += weight
        edges_added += 1
        if edges_added == N - 1:
            break

if uf.is_connected():
    print(mst_weight)
else:
    print(-1)