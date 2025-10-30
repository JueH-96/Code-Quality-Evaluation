import heapq
from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        queries.sort(key=lambda x: x[1])  # Sort by right endpoint
        
        ptr = 0
        pq = []
        diff = [0] * (n + 2)  # Difference array to track coverage increments
        current_coverage = 0
        total_selected = 0
        
        for i in range(n):
            current_coverage += diff[i]  # Update current coverage using the difference array
            
            # Add all queries that can cover the current position i
            while ptr < len(queries) and queries[ptr][1] >= i and queries[ptr][0] <= i:
                r, l = queries[ptr][1], queries[ptr][0]
                heapq.heappush(pq, (r, l))
                ptr += 1
            
            required = nums[i]
            deficit = max(0, required - current_coverage)
            
            while deficit > 0:
                if not pq:
                    return -1  # Not possible to cover all required counts
                r, l = heapq.heappop(pq)
                total_selected += 1
                
                # Update the difference array for this interval's coverage
                diff[l] += 1
                if r + 1 < n:
                    diff[r + 1] -= 1
                
                current_coverage += 1
                deficit -= 1
        
        return len(queries) - total_selected