from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = n // 2
        for k in range(max_k, 0, -1):
            # Check the first k pairs
            s = nums[0] + nums[1]
            valid = True
            for i in range(1, k):
                a = nums[2 * i]
                b = nums[2 * i + 1]
                if a + b != s:
                    valid = False
                    break
            if valid:
                return k
        return 1  # This line is theoretically unreachable due to k >=1