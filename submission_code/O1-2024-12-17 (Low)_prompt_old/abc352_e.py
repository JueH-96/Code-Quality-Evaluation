def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Fast union-find (disjoint set) implementation
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0]*n
            self.count = n  # number of distinct components

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx != ry:
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                self.count -= 1
                return True
            return False
    
    # Parse inputs
    idx = 0
    N = int(input_data[idx]); idx += 1
    M = int(input_data[idx]); idx += 1
    
    edges = []
    # We will build edges in the form (weight, u, v)
    for _ in range(M):
        K_i = int(input_data[idx]); idx += 1
        C_i = int(input_data[idx]); idx += 1
        subset = list(map(lambda x: int(x)-1, input_data[idx: idx+K_i]))
        idx += K_i
        
        # We'll connect subset[0] with all other vertices in the subset
        # with the same weight C_i (K_i - 1 edges).
        root = subset[0]
        for v in subset[1:]:
            edges.append((C_i, root, v))
    
    # If we have fewer than N-1 edges in total, we can skip early,
    # but let's just proceed with MST logic anyway.
    
    # Sort edges by weight
    edges.sort(key=lambda x: x[0])
    
    uf = UnionFind(N)
    mst_weight = 0
    edges_used = 0
    
    # Kruskal's MST
    for w, u, v in edges:
        if uf.union(u, v):
            mst_weight += w
            edges_used += 1
            if edges_used == N-1:
                break
    
    # Check connectivity
    if edges_used < N-1:
        print(-1)
    else:
        print(mst_weight)

# For local testing, comment out the call to solve() if needed.
# But as per the prompt, we call solve() here.
solve()