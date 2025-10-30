import sys
from heapq import heappop, heappush

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def main():
    input = sys.stdin.readline
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        k, c = map(int, input().split())
        vertices = list(map(int, input().split()))
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((c, vertices[i], vertices[j]))

    edges.sort()
    ds = DisjointSet(n)
    total_weight = 0
    for weight, u, v in edges:
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            total_weight += weight

    if ds.find(1) == ds.find(n):
        print(total_weight)
    else:
        print(-1)

if __name__ == "__main__":
    main()