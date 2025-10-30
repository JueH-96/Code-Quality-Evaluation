from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = float('-inf')
        current_sum = 0
        elements = set()
        start = 0
        for end in range(len(nums)):
            # Check if current element is already in the set
            while nums[end] in elements:
                # Remove the element at start from the set and adjust current_sum
                elements.remove(nums[start])
                current_sum -= nums[start]
                start += 1
            # Add the current element to the set and update current_sum
            elements.add(nums[end])
            current_sum += nums[end]
            # Update the max_sum if current_sum is larger
            max_sum = max(max_sum, current_sum)
        return max_sum