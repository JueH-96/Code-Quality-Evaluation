from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        up = [0] * n
        down = [0] * n
        for i in range(n):
            diff = target[i] - nums[i]
            if diff > 0:
                up[i] = diff
                down[i] = 0
            else:
                up[i] = 0
                down[i] = -diff
        
        def calculate(arr):
            res = 0
            prev = 0
            for a in arr:
                if a > prev:
                    res += a - prev
                prev = a
            return res
        
        return calculate(up) + calculate(down)