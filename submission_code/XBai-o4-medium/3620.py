from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        count = 1
        prev = nums[0] - k
        for i in range(1, n):
            current = nums[i]
            new_val = max(prev + 1, current - k)
            if new_val <= current + k:
                count += 1
                prev = new_val
            else:
                pass  # Cannot adjust to make distinct, skip
        return count