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
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def kruskal(edges, n):
    uf = UnionFind(n)
    mst_weight = 0
    for weight, u, v in sorted(edges):
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_weight += weight
    return mst_weight

def main():
    input_data = sys.stdin.read().splitlines()
    n, m = map(int, input_data[0].split())
    edges = []
    line_index = 1

    for _ in range(m):
        k, c = map(int, input_data[line_index].split())
        vertices = list(map(int, input_data[line_index + 1].split()))
        line_index += 2

        for i in range(k):
            for j in range(i + 1, k):
                edges.append((c, vertices[i] - 1, vertices[j] - 1))

    # Check if the graph is connected
    adj_list = defaultdict(set)
    for _, u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)

    visited = set()

    def dfs(v):
        visited.add(v)
        for neighbor in adj_list[v]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)

    if len(visited) != n:
        print(-1)
    else:
        print(kruskal(edges, n))

if __name__ == "__main__":
    main()