from typing import List
from math import gcd

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        n = len(nums)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if lcm(nums[px], nums[py]) <= threshold:
                    parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)
        
        return sum(parent[i] == i for i in range(n))