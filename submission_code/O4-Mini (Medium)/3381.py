from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        best = float('inf')
        
        for i in range(n):
            curr_or = 0
            # Try all subarrays starting at i
            for j in range(i, n):
                curr_or |= nums[j]
                # As soon as OR >= k, record length and break
                if curr_or >= k:
                    best = min(best, j - i + 1)
                    break
        
        return best if best != float('inf') else -1