from typing import List
from math import gcd

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)

        class UnionFind:
            def __init__(self, size: int):
                self.parent = list(range(size))
                self.rank = [1] * size

            def find(self, x: int) -> int:
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x: int, y: int) -> None:
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

        n = len(nums)
        uf = UnionFind(n)

        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    uf.union(i, j)

        # Count the number of unique roots
        unique_roots = set()
        for i in range(n):
            unique_roots.add(uf.find(i))

        return len(unique_roots)