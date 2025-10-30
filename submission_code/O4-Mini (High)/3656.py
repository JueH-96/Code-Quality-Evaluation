from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # Maximum possible operations to clear the array
        max_ops = (n + 2) // 3
        
        for k in range(max_ops + 1):
            start = 3 * k
            # If we've removed at least n elements, array is empty → distinct
            if start >= n:
                return k
            # Check if the suffix from 'start' to end has all distinct elements
            suffix = nums[start:]
            if len(suffix) == len(set(suffix)):
                return k
        
        # Fallback (should not actually be reached)
        return max_ops