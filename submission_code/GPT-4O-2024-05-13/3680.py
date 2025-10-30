from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        n = len(nums)
        
        # Union-Find data structure
        parent = list(range(n))
        rank = [1] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                if rank[rootX] > rank[rootY]:
                    parent[rootY] = rootX
                elif rank[rootX] < rank[rootY]:
                    parent[rootX] = rootY
                else:
                    parent[rootY] = rootX
                    rank[rootX] += 1
        
        # Build the graph by connecting nodes with lcm <= threshold
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)
        
        # Count the number of unique connected components
        unique_components = len(set(find(i) for i in range(n)))
        return unique_components