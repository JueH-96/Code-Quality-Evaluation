class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
        
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
        return True

def solve():
    # Read input
    N, M = map(int, input().split())
    
    # If no edges, already a forest
    if M == 0:
        return 0
    
    # Create Union-Find data structure
    uf = UnionFind(N+1)  # +1 because vertices are 1-indexed
    
    # Count edges that create cycles
    edges_to_delete = 0
    
    # Process each edge
    for _ in range(M):
        u, v = map(int, input().split())
        # If vertices are already connected, this edge creates a cycle
        if not uf.union(u, v):
            edges_to_delete += 1
            
    return edges_to_delete

# Print the answer
print(solve())