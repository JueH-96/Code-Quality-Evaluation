from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        seen = set()
        left = 0
        current_sum = 0
        max_sum = float('-inf')

        for right in range(n):
            # If nums[right] is already in the window, shrink from the left
            while nums[right] in seen:
                seen.remove(nums[left])
                current_sum -= nums[left]
                left += 1
            # Add the new unique element
            seen.add(nums[right])
            current_sum += nums[right]
            # Update the maximum sum seen so far
            max_sum = max(max_sum, current_sum)

        return max_sum