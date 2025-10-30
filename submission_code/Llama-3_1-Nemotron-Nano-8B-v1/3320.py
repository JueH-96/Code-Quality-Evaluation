from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        max_k = len(nums) // 2
        for k in range(max_k, 0, -1):
            target = nums[0] + nums[1]
            valid = True
            for i in range(k):
                a = 2 * i
                b = 2 * i + 1
                if b >= len(nums) or nums[a] + nums[b] != target:
                    valid = False
                    break
            if valid:
                return k
        return 0