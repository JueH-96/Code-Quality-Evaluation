import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
            return True
        return False

def kruskal(edges, n):
    uf = UnionFind(n)
    mst_cost = 0
    for cost, u, v in sorted(edges):
        if uf.union(u, v):
            mst_cost += cost
    return mst_cost

def main():
    input_data = sys.stdin.read().splitlines()
    n, q = map(int, input_data[0].split())
    edges = []
    connected = [False] * (n + 1)
    
    for i in range(1, q + 1):
        l, r, c = map(int, input_data[i].split())
        for j in range(l, r + 1):
            edges.append((c, n + i, j))
            connected[j] = True
    
    if all(connected[1:]):
        mst_cost = kruskal(edges, n + q + 1)
        print(mst_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()