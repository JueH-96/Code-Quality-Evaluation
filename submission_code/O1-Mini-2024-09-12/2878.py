from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        ongoing = 0

        for i in range(n):
            ongoing += diff[i]
            need = nums[i] - ongoing
            if need < 0:
                return False
            if need > 0:
                if i + k > n:
                    return False
                diff[i] += need
                diff[i + k] -= need
                ongoing += need

        return True