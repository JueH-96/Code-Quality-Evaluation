from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            distinct_count = len(set(subarray))
            if distinct_count >= m:
                subarray_sum = sum(subarray)
                max_sum = max(max_sum, subarray_sum)
        return max_sum