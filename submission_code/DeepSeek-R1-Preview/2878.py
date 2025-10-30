from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        end_ops = [0] * (n + 1)
        current_ops = 0
        for i in range(n):
            current_ops -= end_ops[i]
            if current_ops > nums[i]:
                return False
            required = nums[i] - current_ops
            if required < 0:
                return False
            if required > 0:
                if i + k > n:
                    return False
                current_ops += required
                end_ops[i + k] += required
        return True