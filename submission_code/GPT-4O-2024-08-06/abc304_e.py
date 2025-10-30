class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        rootU = self.find(u)
        rootV = self.find(v)
        if rootU != rootV:
            if self.rank[rootU] > self.rank[rootV]:
                self.parent[rootV] = rootU
            elif self.rank[rootU] < self.rank[rootV]:
                self.parent[rootU] = rootV
            else:
                self.parent[rootV] = rootU
                self.rank[rootU] += 1
            return True
        return False

import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    uf = UnionFind(N)
    
    for _ in range(M):
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1
        uf.union(u, v)
    
    K = int(data[index])
    index += 1
    
    forbidden_pairs = []
    for _ in range(K):
        x = int(data[index]) - 1
        index += 1
        y = int(data[index]) - 1
        index += 1
        forbidden_pairs.append((x, y))
    
    Q = int(data[index])
    index += 1
    
    results = []
    for _ in range(Q):
        p = int(data[index]) - 1
        index += 1
        q = int(data[index]) - 1
        index += 1
        
        # Temporarily add the edge (p, q)
        original_p_root = uf.find(p)
        original_q_root = uf.find(q)
        uf.union(p, q)
        
        # Check if any forbidden pair is now connected
        is_good = True
        for x, y in forbidden_pairs:
            if uf.find(x) == uf.find(y):
                is_good = False
                break
        
        if is_good:
            results.append("Yes")
        else:
            results.append("No")
        
        # Revert the union operation
        uf.parent[original_q_root] = original_q_root
        uf.parent[original_p_root] = original_p_root
        if uf.rank[original_p_root] == uf.rank[original_q_root]:
            uf.rank[original_p_root] -= 1
    
    print("
".join(results))

main()