from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            and_value = nums[i]
            for j in range(i, n):
                and_value &= nums[j]
                if and_value == k:
                    count += 1
                elif and_value < k:
                    break
        return count