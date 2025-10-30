from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        # Remove duplicates and sort the list
        nums = list(set(nums))
        if 1 in nums:
            return False
        
        # Find all prime factors of each number
        def prime_factors(n):
            i = 2
            factors = []
            while i * i <= n:
                if n % i:
                    i += 1
                else:
                    n //= i
                    factors.append(i)
            if n > 1:
                factors.append(n)
            return factors
        
        # Create a graph where nodes are indices and edges are gcd > 1
        graph = defaultdict(list)
        factor_to_indices = defaultdict(list)
        
        for i, num in enumerate(nums):
            factors = prime_factors(num)
            for factor in factors:
                for j in factor_to_indices[factor]:
                    graph[i].append(j)
                    graph[j].append(i)
                factor_to_indices[factor].append(i)
        
        # Perform DFS to check if all nodes are connected
        def dfs(node, visited):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, visited)
        
        visited = set()
        dfs(0, visited)
        
        return len(visited) == len(nums)