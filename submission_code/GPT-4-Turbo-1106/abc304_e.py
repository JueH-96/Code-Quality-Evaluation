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

def main():
    input_lines = sys.stdin.readlines()
    n, m = map(int, input_lines[0].split())
    edges = [tuple(map(int, line.split())) for line in input_lines[1:m+1]]
    k = int(input_lines[m+1])
    forbidden_paths = [tuple(map(int, line.split())) for line in input_lines[m+2:m+2+k]]
    q = int(input_lines[m+2+k])
    queries = [tuple(map(int, line.split())) for line in input_lines[m+3+k:]]

    # Create a union-find structure to keep track of connected components
    uf = UnionFind(n + 1)  # +1 because vertices are 1-indexed

    # Union all connected vertices
    for u, v in edges:
        uf.union(u, v)

    # Create a set of forbidden connections
    forbidden_connections = set()
    for x, y in forbidden_paths:
        forbidden_connections.add((uf.find(x), uf.find(y)))

    # Process queries
    for p, q in queries:
        if (uf.find(p), uf.find(q)) in forbidden_connections or (uf.find(q), uf.find(p)) in forbidden_connections:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()