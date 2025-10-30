from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        current_ops = 0
        delta_ops = [0] * (n + 1)
        for i in range(n):
            current_ops += delta_ops[i]
            if i >= n - k + 1:
                if nums[i] != current_ops:
                    return False
                continue
            required = nums[i] - current_ops
            if required < 0:
                return False
            current_ops += required
            if i + k < n:
                delta_ops[i + k] -= required
        return True