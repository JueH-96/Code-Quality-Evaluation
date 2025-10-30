from typing import List
import heapq

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        req = nums.copy()
        
        # Check initial coverage using prefix sum
        coverage = [0] * (n + 1)
        for l, r in queries:
            coverage[l] += 1
            if r + 1 < n:
                coverage[r + 1] -= 1
        for i in range(n):
            coverage[i] = coverage[i] + (coverage[i-1] if i > 0 else 0)
            if coverage[i] < req[i]:
                return -1
        
        # Sort queries by their right endpoint
        queries.sort(key=lambda x: x[1])
        max_req = max(req)
        total_selected = 0
        
        for layer in range(1, max_req + 1):
            current_pos = 0
            selected = 0
            ptr = 0
            while current_pos < n:
                # Find the best interval that covers current_pos and extends the furthest
                max_r = -1
                best_idx = -1
                while ptr < m and queries[ptr][0] <= current_pos:
                    l, r = queries[ptr]
                    if r > max_r:
                        max_r = r
                        best_idx = ptr
                    ptr += 1
                if best_idx == -1:
                    break  # This should not happen due to initial check
                selected += 1
                current_pos = max_r + 1
            total_selected += selected
        
        return len(queries) - total_selected