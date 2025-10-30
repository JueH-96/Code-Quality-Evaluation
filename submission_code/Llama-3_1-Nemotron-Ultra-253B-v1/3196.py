from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_freq = 0
        left = 0
        for right in range(n):
            while True:
                mid = (left + right) // 2
                # Calculate the cost to make all elements in [left, right] equal to nums[mid]
                cost = nums[mid] * (mid - left + 1) - (prefix[mid + 1] - prefix[left])
                cost += (prefix[right + 1] - prefix[mid + 1] - nums[mid] * (right - mid))
                if cost <= k:
                    break
                left += 1
            max_freq = max(max_freq, right - left + 1)
        return max_freq