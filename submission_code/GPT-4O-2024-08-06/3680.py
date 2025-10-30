from typing import List
from math import gcd

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Helper function to find the least common multiple
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        # Union-Find (Disjoint Set Union) data structure
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [1] * size

            def find(self, u):
                if self.parent[u] != u:
                    self.parent[u] = self.find(self.parent[u])
                return self.parent[u]

            def union(self, u, v):
                rootU = self.find(u)
                rootV = self.find(v)
                if rootU != rootV:
                    if self.rank[rootU] > self.rank[rootV]:
                        self.parent[rootV] = rootU
                    elif self.rank[rootU] < self.rank[rootV]:
                        self.parent[rootU] = rootV
                    else:
                        self.parent[rootV] = rootU
                        self.rank[rootU] += 1

        n = len(nums)
        uf = UnionFind(n)

        # Check all pairs (i, j) and union them if lcm(nums[i], nums[j]) <= threshold
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    uf.union(i, j)

        # Count unique roots to determine the number of connected components
        unique_roots = set()
        for i in range(n):
            unique_roots.add(uf.find(i))

        return len(unique_roots)