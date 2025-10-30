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

def solve():
    n, k = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Build connected components
    uf = UnionFind(n)
    for i in range(n):
        for j in range(max(0, i-k), min(n, i+k+1)):
            if i != j:
                uf.union(i, j)
    
    # Group positions by component
    components = {}
    for i in range(n):
        root = uf.find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)
    
    # Check each component
    for positions in components.values():
        # Collect available values in this component
        available = set()
        for pos in positions:
            available.add(A[pos])
        
        # Check if all required values are available
        for pos in positions:
            if B[pos] not in available:
                return "No"
    
    return "Yes"

T = int(input())
for _ in range(T):
    print(solve())