import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size + 1))
        self.rank = [1] * (size + 1)
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1
        return True

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    uf = UnionFind(N)
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        idx += 2
        uf.union(u, v)
    K = int(data[idx])
    idx += 1
    forbidden = []
    for _ in range(K):
        x = int(data[idx])
        y = int(data[idx+1])
        idx += 2
        forbidden.append((x, y))
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        p = int(data[idx])
        q = int(data[idx+1])
        idx += 2
        queries.append((p, q))
    # Precompute the roots for all forbidden pairs
    forbidden_roots = set()
    for x, y in forbidden:
        root_x = uf.find(x)
        root_y = uf.find(y)
        forbidden_roots.add((root_x, root_y))
        forbidden_roots.add((root_y, root_x))
    # Process each query
    for p, q in queries:
        root_p = uf.find(p)
        root_q = uf.find(q)
        if (root_p, root_q) in forbidden_roots or (root_q, root_p) in forbidden_roots:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()