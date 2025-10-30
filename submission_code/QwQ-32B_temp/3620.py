from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        if not nums:
            return 0
        count = 1
        prev = nums[0] - k
        for i in range(1, len(nums)):
            current = nums[i]
            current_min = current - k
            required = prev + 1
            adjusted = max(current_min, required)
            current_max = current + k
            if adjusted <= current_max:
                prev = adjusted
                count += 1
        return count