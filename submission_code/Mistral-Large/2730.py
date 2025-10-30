from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        # Sort the array in descending order
        nums.sort(reverse=True)

        # Initialize the maximum OR value
        max_or = 0

        # Apply the operation k times
        for i in range(k):
            # Multiply the largest element by 2
            nums[0] *= 2
            # Update the maximum OR value
            max_or = nums[0]
            for num in nums[1:]:
                max_or |= num

        return max_or