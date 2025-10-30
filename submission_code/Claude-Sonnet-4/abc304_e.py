class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def copy(self):
        new_uf = UnionFind(len(self.parent))
        # Normalize all parents first
        for i in range(len(self.parent)):
            self.find(i)
        # Copy the normalized structure
        new_uf.parent = self.parent[:]
        new_uf.rank = self.rank[:]
        return new_uf

# Read input
N, M = map(int, input().split())

# Store edges to rebuild graph for each query
edges = []
for _ in range(M):
    u, v = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1
    edges.append((u, v))

K = int(input())
forbidden_pairs = []
for _ in range(K):
    x, y = map(int, input().split())
    x -= 1  # Convert to 0-indexed
    y -= 1
    forbidden_pairs.append((x, y))

Q = int(input())
for _ in range(Q):
    p, q = map(int, input().split())
    p -= 1  # Convert to 0-indexed
    q -= 1
    
    # Create new Union-Find and build graph with new edge
    uf = UnionFind(N)
    
    # Add original edges
    for u, v in edges:
        uf.union(u, v)
    
    # Add the new edge
    uf.union(p, q)
    
    # Check if any forbidden pair becomes connected
    is_good = True
    for x, y in forbidden_pairs:
        if uf.connected(x, y):
            is_good = False
            break
    
    print("Yes" if is_good else "No")