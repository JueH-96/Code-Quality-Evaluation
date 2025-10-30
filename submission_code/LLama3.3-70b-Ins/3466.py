from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)):
            and_val = nums[i]
            for j in range(i, len(nums)):
                and_val &= nums[j]
                if and_val == k:
                    count += 1
        return count