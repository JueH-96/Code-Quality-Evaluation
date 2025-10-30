from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        # If k is 0, any non-empty subarray has OR >= 0
        if k == 0:
            return 1
        
        n = len(nums)
        best = float('inf')
        
        # Try all starting positions
        for i in range(n):
            current_or = 0
            # Extend the subarray from i to j
            for j in range(i, n):
                current_or |= nums[j]
                # Once we reach OR >= k, we can update the answer and break
                if current_or >= k:
                    best = min(best, j - i + 1)
                    break
        
        return best if best != float('inf') else -1