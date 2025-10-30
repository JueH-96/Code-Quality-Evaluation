from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = n // 2

        for k in range(max_k, 0, -1):
            s = nums[0] + nums[1]
            valid = True
            for i in range(k):
                if nums[2 * i] + nums[2 * i + 1] != s:
                    valid = False
                    break
            if valid:
                return k
        return 0