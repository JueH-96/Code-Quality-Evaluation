import sys
from heapq import heappop, heappush

def main():
    input = sys.stdin.read
    data = input().split()
    N, Q = int(data[0]), int(data[1])
    LRC = [(int(data[i]), int(data[i+1]), int(data[i+2])) for i in range(2, 3*Q, 3)]
    
    edges = []
    for i, (L, R, C) in enumerate(LRC):
        edges.append((C, N + i + 1, L))
        edges.append((C, N + i + 1, R + 1))
    
    edges.sort()
    uf = UnionFind(N + Q)
    min_cost = 0
    used_edges = 0
    
    for cost, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            min_cost += cost
            used_edges += 1
    
    if used_edges == N + Q - 1:
        print(min_cost)
    else:
        print(-1)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

if __name__ == "__main__":
    main()