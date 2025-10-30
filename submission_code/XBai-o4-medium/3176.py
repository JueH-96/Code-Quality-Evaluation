from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if nums[i] < nums[j] and nums[k] < nums[j]:
                        current_sum = nums[i] + nums[j] + nums[k]
                        if current_sum < min_sum:
                            min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1