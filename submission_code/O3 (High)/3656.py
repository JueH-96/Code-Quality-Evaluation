from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        # After k operations we will have removed exactly 3*k elements
        # (as long as 3*k ≤ n).  The remaining array is nums[3*k:].
        # We search for the smallest k for which this suffix is already distinct.
        for k in range(0, n // 3 + 1):
            suffix = nums[3 * k:]
            if len(suffix) == len(set(suffix)):
                return k
        
        # If no non-empty reachable suffix is distinct, we must delete
        # everything.  That takes ceil(n/3) operations.
        return (n + 2) // 3