from typing import List
import collections

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def maxSubarraySum(arr):
            max_sum = float('-inf')
            current_sum = 0
            for num in arr:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum

        original_max = maxSubarraySum(nums)
        counter = collections.Counter(nums)

        for x in counter:
            if counter[x] == len(nums):
                continue
            new_array = [num for num in nums if num != x]
            original_max = max(original_max, maxSubarraySum(new_array))

        return original_max