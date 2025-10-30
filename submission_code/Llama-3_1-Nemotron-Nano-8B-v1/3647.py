from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # Compute the coverage for each position using the prefix sum method
        coverage = [0] * (n + 1)
        for l, r in queries:
            coverage[l] += 1
            if r + 1 < n:
                coverage[r + 1] -= 1
        
        # Compute the actual coverage for each position and check feasibility
        current = 0
        for i in range(n):
            current += coverage[i]
            if current < nums[i]:
                return -1
        
        # Compute sum_c, max_c, and max_coverage
        sum_c = sum(nums)
        max_c = max(nums)
        max_coverage = max(r - l + 1 for l, r in queries)
        
        # Calculate the minimal number of queries required
        minimal_queries = max(max_c, (sum_c + max_coverage - 1) // max_coverage)
        
        # The answer is total_queries - minimal_queries
        return len(queries) - minimal_queries