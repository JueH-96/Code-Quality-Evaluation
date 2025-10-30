from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        left = 0
        total_ops = 0
        max_freq = 0

        for right in range(n):
            # Calculate the number of operations needed to make all elements
            # from nums[left] to nums[right] equal to nums[right]
            total_ops += (right - left + 1) * (nums[right] - nums[right - 1]) if right > 0 else 0

            # If the total operations exceed k, move the left pointer to reduce the window size
            while total_ops > k:
                total_ops -= (right - left) * (nums[left + 1] - nums[left]) if left + 1 < right else 0
                left += 1

            # The current window size is the maximum frequency we can achieve
            max_freq = max(max_freq, right - left + 1)

        return max_freq