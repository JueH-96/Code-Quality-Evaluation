# YOUR CODE HERE
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = {i: [i] for i in range(n)}
    
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
        
        # Merge components
        self.components[px].extend(self.components[py])
        self.components[px].sort(reverse=True)
        del self.components[py]
    
    def get_kth_largest(self, x, k):
        root = self.find(x)
        if k > len(self.components[root]):
            return -1
        return self.components[root][k-1] + 1  # Convert back to 1-indexed

n, q = map(int, input().split())
uf = UnionFind(n)

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        u, v = query[1] - 1, query[2] - 1  # Convert to 0-indexed
        uf.union(u, v)
    else:
        v, k = query[1] - 1, query[2]  # Convert v to 0-indexed
        print(uf.get_kth_largest(v, k))