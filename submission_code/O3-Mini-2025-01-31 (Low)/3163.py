from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        # Generate all subarrays
        for i in range(n):
            # Use a set to track distinct elements in the current subarray
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                count = len(seen)
                total += count * count  # add square of the distinct count
        return total