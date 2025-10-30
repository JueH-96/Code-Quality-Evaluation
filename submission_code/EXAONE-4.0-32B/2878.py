from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        curr = 0
        for i in range(n):
            curr += diff[i]
            if nums[i] < curr:
                return False
            if i <= n - k:
                extra = nums[i] - curr
                curr += extra
                if i + k < n:
                    diff[i + k] -= extra
            else:
                if nums[i] != curr:
                    return False
        return True