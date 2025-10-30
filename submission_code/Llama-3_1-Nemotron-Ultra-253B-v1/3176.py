from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            left = [nums[i] for i in range(j) if nums[i] < nums[j]]
            if not left:
                continue
            left_min = min(left)
            right = [nums[k] for k in range(j + 1, n) if nums[k] < nums[j]]
            if not right:
                continue
            right_min = min(right)
            current_sum = left_min + nums[j] + right_min
            if current_sum < min_sum:
                min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1