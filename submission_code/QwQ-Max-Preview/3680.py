import math
from typing import List

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
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
    def countComponents(self, nums: List[int], threshold: int) -> int:
        num_set = set(nums)
        index_map = {num: i for i, num in enumerate(nums)}
        n = len(nums)
        dsu = DSU(n)
        
        for m in range(1, threshold + 1):
            divisors = set()
            for i in range(1, int(math.isqrt(m)) + 1):
                if m % i == 0:
                    divisors.add(i)
                    divisors.add(m // i)
            present = []
            for d in divisors:
                if d in num_set:
                    present.append(d)
            if len(present) < 2:
                continue
            indices = [index_map[x] for x in present]
            for i in range(1, len(indices)):
                dsu.union(indices[i-1], indices[i])
        
        roots = set()
        for i in range(n):
            roots.add(dsu.find(i))
        return len(roots)