import sys
from collections import defaultdict, deque

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
        fx = self.find(x)
        fy = self.find(y)
        if fx == fy:
            return False
        if self.rank[fx] < self.rank[fy]:
            self.parent[fx] = fy
        else:
            self.parent[fy] = fx
            if self.rank[fx] == self.rank[fy]:
                self.rank[fx] += 1
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
    # Precompute the forbidden pairs' roots
    forbidden_roots = set()
    for x, y in forbidden:
        fx = uf.find(x)
        fy = uf.find(y)
        forbidden_roots.add((fx, fy))
        forbidden_roots.add((fy, fx))
    # Process each query
    for p, q in queries:
        fp = uf.find(p)
        fq = uf.find(q)
        if (fp, fq) in forbidden_roots or (fq, fp) in forbidden_roots:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()