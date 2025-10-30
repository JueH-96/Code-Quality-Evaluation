from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            # Find the minimum value to the left of j that is less than nums[j]
            left_min = float('inf')
            for i in range(j):
                if nums[i] < nums[j] and nums[i] < left_min:
                    left_min = nums[i]
            # Find the minimum value to the right of j that is less than nums[j]
            right_min = float('inf')
            for k in range(j + 1, n):
                if nums[k] < nums[j] and nums[k] < right_min:
                    right_min = nums[k]
            # Check if both left and right minima exist
            if left_min != float('inf') and right_min != float('inf'):
                current_sum = left_min + nums[j] + right_min
                if current_sum < min_sum:
                    min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1