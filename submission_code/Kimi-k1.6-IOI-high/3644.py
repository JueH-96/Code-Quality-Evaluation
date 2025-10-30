from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        n = len(nums)
        for k in range(l, r + 1):
            for i in range(n - k + 1):
                current_sum = sum(nums[i:i+k])
                if current_sum > 0 and current_sum < min_sum:
                    min_sum = current_sum
        return min_sum if min_sum != float('inf') else -1