from typing import List

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
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
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        if n == 0:
            return 0
        
        sets = [set(prop) for prop in properties]
        dsu = DSU(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                common = len(sets[i] & sets[j])
                if common >= k:
                    dsu.union(i, j)
        
        roots = set()
        for i in range(n):
            roots.add(dsu.find(i))
        
        return len(roots)