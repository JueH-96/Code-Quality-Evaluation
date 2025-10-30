from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # We can do at most ceil(n/3) operations to clear the entire array.
        max_ops = (n + 2) // 3  # smallest integer >= n/3
        for k in range(max_ops + 1):
            # After k operations, we've removed first 3*k elements (or all if 3*k >= n)
            start = 3 * k
            if start >= n:
                # Array is empty
                return k
            suffix = nums[start:]
            # Check if suffix has all distinct elements
            if len(suffix) == len(set(suffix)):
                return k
        # In principle, we should always return by the loop above
        return max_ops