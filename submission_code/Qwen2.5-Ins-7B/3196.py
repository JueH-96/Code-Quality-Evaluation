from typing import List
from collections import defaultdict

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        max_freq = 0
        left = 0
        for right in range(n):
            target = nums[right]
            while nums[left] < target - (prefix_sum[right + 1] - prefix_sum[left]):
                left += 1
            max_freq = max(max_freq, right - left + 1)
            k -= (right - left) * (target - nums[left])
            if k < 0:
                k += (nums[left + 1] - nums[left]) * (left + 1)
                left += 1
        
        return max_freq