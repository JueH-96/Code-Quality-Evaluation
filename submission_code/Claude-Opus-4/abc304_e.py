# YOUR CODE HERE
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

# Read input
N, M = map(int, input().split())

# Build the graph using Union-Find
uf = UnionFind(N)
for _ in range(M):
    u, v = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1
    uf.union(u, v)

# Read forbidden pairs
K = int(input())
forbidden_pairs = []
for _ in range(K):
    x, y = map(int, input().split())
    x -= 1  # Convert to 0-indexed
    y -= 1
    forbidden_pairs.append((x, y))

# Process queries
Q = int(input())
for _ in range(Q):
    p, q = map(int, input().split())
    p -= 1  # Convert to 0-indexed
    q -= 1
    
    # Check if adding edge (p, q) would violate any forbidden pair
    is_good = True
    
    # Get components of p and q
    comp_p = uf.find(p)
    comp_q = uf.find(q)
    
    # If p and q are already in the same component, adding edge doesn't change anything
    if comp_p == comp_q:
        print("Yes")
        continue
    
    # Check each forbidden pair
    for x, y in forbidden_pairs:
        comp_x = uf.find(x)
        comp_y = uf.find(y)
        
        # Check if the new edge would connect x and y
        # This happens if one is in comp_p and the other is in comp_q
        if (comp_x == comp_p and comp_y == comp_q) or (comp_x == comp_q and comp_y == comp_p):
            is_good = False
            break
    
    if is_good:
        print("Yes")
    else:
        print("No")