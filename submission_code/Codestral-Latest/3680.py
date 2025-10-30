from math import gcd
from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)

        n = len(nums)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootX] = rootY

        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)

        for i in range(n):
            find(i)

        return len(set(parent))