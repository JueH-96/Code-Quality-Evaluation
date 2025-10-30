import heapq
from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        total_queries = len(queries)
        
        starts = [[] for _ in range(n)]
        for l, r in queries:
            if l < n:
                starts[l].append(r)
        
        available = []
        active_ends = []
        used = 0
        
        for i in range(n):
            for r_val in starts[i]:
                heapq.heappush(available, -r_val)
            
            while active_ends and active_ends[0] < i:
                heapq.heappop(active_ends)
            
            current_coverage = len(active_ends)
            
            while current_coverage < nums[i]:
                if not available:
                    return -1
                neg_r = heapq.heappop(available)
                r_val = -neg_r
                heapq.heappush(active_ends, r_val)
                current_coverage += 1
                used += 1
                
        return total_queries - used