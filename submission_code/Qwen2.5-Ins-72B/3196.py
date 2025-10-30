from typing import List
from collections import Counter

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        def cost(start, end):
            mid = (start + end) // 2
            left_cost = (prefix_sum[mid] - prefix_sum[start]) - (mid - start) * nums[mid]
            right_cost = (end - mid) * nums[mid] - (prefix_sum[end] - prefix_sum[mid])
            return left_cost + right_cost
        
        max_freq = 1
        left = 0
        for right in range(n):
            while left < right and cost(left, right + 1) > k:
                left += 1
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq