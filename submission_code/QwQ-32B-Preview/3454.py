from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        total_ops = 0
        i = 0
        while i < n:
            while i < n and nums[i] == target[i]:
                i += 1
            if i == n:
                break
            # Start of a new run
            current_sign = 1 if nums[i] < target[i] else -1
            max_diff = abs(nums[i] - target[i])
            i += 1
            while i < n and nums[i] != target[i] and (nums[i] - target[i]) * current_sign > 0:
                max_diff = max(max_diff, abs(nums[i] - target[i]))
                i += 1
            total_ops += max_diff
        return total_ops