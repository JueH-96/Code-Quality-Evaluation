from math import gcd
from collections import defaultdict
from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX

        n = len(nums)
        parent = list(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if gcd(nums[i], nums[j]) > 1:
                    union(i, j)

        # Check if all elements are in the same connected component
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False

        return True