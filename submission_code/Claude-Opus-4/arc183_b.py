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
    T = int(input())
    
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        
        # Create union-find structure
        uf = UnionFind(N)
        
        # Union positions that can reach each other
        for i in range(N):
            for j in range(max(0, i - K), min(N, i + K + 1)):
                uf.union(i, j)
        
        # Group positions by their root
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(N):
            groups[uf.find(i)].append(i)
        
        # Check each group
        possible = True
        for positions in groups.values():
            # Count values in A and B for this group
            a_values = {}
            b_values = {}
            
            for pos in positions:
                a_val = A[pos]
                b_val = B[pos]
                
                a_values[a_val] = a_values.get(a_val, 0) + 1
                b_values[b_val] = b_values.get(b_val, 0) + 1
            
            # Check if multisets match
            if a_values != b_values:
                possible = False
                break
        
        print("Yes" if possible else "No")

solve()