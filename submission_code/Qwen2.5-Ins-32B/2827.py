from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False
        
        def prime_factors(n):
            i = 2
            factors = set()
            while i * i <= n:
                if n % i:
                    i += 1
                else:
                    n //= i
                    factors.add(i)
            if n > 1:
                factors.add(n)
            return factors
        
        factor_to_indices = defaultdict(list)
        for i, num in enumerate(nums):
            for factor in prime_factors(num):
                factor_to_indices[factor].append(i)
        
        def dfs(node, visited):
            if node in visited:
                return
            visited.add(node)
            for neighbor in factor_to_indices[nums[node]]:
                dfs(neighbor, visited)
        
        visited = set()
        dfs(0, visited)
        return len(visited) == len(nums)