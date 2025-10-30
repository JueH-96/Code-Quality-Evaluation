import sys
from sys import stdin
from collections import defaultdict

class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))
        self.rank = [1]*(n+1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
            self.rank[y_root] += self.rank[x_root]
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += self.rank[y_root]

def main():
    n, m = map(int, stdin.readline().split())
    dsu = DSU(n)
    edges_count = defaultdict(int)
    for _ in range(m):
        a, b = map(int, stdin.readline().split())
        ra = dsu.find(a)
        rb = dsu.find(b)
        if ra != rb:
            dsu.union(ra, rb)
        root = dsu.find(a)
        edges_count[root] += 1
    ans = 0
    for cnt in edges_count.values():
        ans += cnt * (cnt -1) // 2
    print(ans)

if __name__ == "__main__":
    main()