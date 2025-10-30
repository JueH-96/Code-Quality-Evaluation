from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # Maximum operations needed would be when the array becomes empty.
        # In each operation, we remove 3 elements.
        max_ops = (n + 2) // 3  # Ceiling of n/3
        for ops in range(max_ops + 1):
            remaining = nums[ops * 3:]
            # Check if the remaining array has distinct elements
            if len(remaining) == len(set(remaining)):
                return ops
        return max_ops  # This line is never expected to be reached.