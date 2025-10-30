from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # Try removing k operations (i.e., removing first 3*k elements)
        max_ops = (n + 2) // 3  # ceil(n/3)
        for k in range(max_ops + 1):
            # Compute start index of the remaining suffix
            cut = min(n, 3 * k)
            suffix = nums[cut:]
            # If the suffix has all distinct elements, we're done
            if len(set(suffix)) == len(suffix):
                return k
        # In the worst case, we remove everything
        return max_ops