from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        for i in range(len(nums)):
            cnt = 0
            for j in range(i, len(nums)):
                if nums[j] % modulo == k:
                    cnt += 1
                if cnt % modulo == k:
                    count += 1
        return count