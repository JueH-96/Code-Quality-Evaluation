import math
from typing import List

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        parent = list(range(n))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        def lcm(a, b):
            return (a * b) // math.gcd(a, b)

        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)

        components = set()
        for i in range(n):
            components.add(find(i))
        
        return len(components)