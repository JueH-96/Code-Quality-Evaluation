from functools import lru_cache

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        @lru_cache(None)
        def dp(index, selected, or1, or2):
            # Base case: selected 2k elements
            if selected == 2 * k:
                return or1 ^ or2
            
            # No more elements to select
            if index == n:
                return -float('inf')
            
            # Option 1: Skip current element
            res = dp(index + 1, selected, or1, or2)
            
            # Option 2: Select current element
            if selected < k:
                # Add to first half
                res = max(res, dp(index + 1, selected + 1, or1 | nums[index], or2))
            else:
                # Add to second half
                res = max(res, dp(index + 1, selected + 1, or1, or2 | nums[index]))
            
            return res
        
        return dp(0, 0, 0, 0)