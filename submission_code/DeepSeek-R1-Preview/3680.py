from typing import List
import bisect

class UnionFind:
    def __init__(self, elements):
        self.parent = {}
        for e in elements:
            self.parent[e] = e
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        nums = list(set(nums))  # Ensure unique elements
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        uf = UnionFind(nums)
        S = set(nums)
        sorted_nums = nums
        
        for i in range(n):
            x = nums[i]
            if x > threshold:
                continue
            
            # Connect multiples of x
            k = 2
            while True:
                y = x * k
                if y > threshold:
                    break
                if y in S:
                    uf.union(x, y)
                k += 1
            
            # Connect y's where y <= threshold // x
            max_y = threshold // x
            idx = bisect.bisect_right(sorted_nums, max_y)
            for j in range(idx):
                y = sorted_nums[j]
                if y != x:
                    uf.union(x, y)
        
        roots = set()
        for x in nums:
            if x > threshold:
                roots.add(x)
            else:
                roots.add(uf.find(x))
        
        return len(roots)