from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Sort the array 'nums' in non-decreasing order.
        # This is the first crucial step. If a set of numbers can be transformed
        # into a common value X, it means the maximum value in that set minus
        # the minimum value in that set must be at most 2k.
        # Sorting allows us to efficiently find such a set using a sliding window.
        nums.sort()

        n = len(nums)
        # Initialize 'max_beauty' to 0. Since constraints state nums.length >= 1,
        # the beauty will be at least 1 for a single element.
        max_beauty = 0
        
        # 'left' pointer marks the beginning of our current window.
        left = 0
        
        # The 'right' pointer iterates through the array, expanding the window.
        for right in range(n):
            # The condition for all elements within the window nums[left...right]
            # to be transformable into a common value is: nums[right] - nums[left] <= 2k.
            # (nums[right] is the maximum and nums[left] is the minimum in a sorted window).
            # If this condition is violated (i.e., nums[right] - nums[left] > 2k),
            # it means the current window is too wide. We need to shrink it from the left
            # by incrementing 'left' until the condition is satisfied again.
            while nums[right] - nums[left] > 2 * k:
                left += 1
            
            # At this point, the current window nums[left...right] is valid.
            # All elements within this window can be transformed to a common value.
            # Calculate the length of this valid window (right - left + 1)
            # and update 'max_beauty' if this window is longer than any found so far.
            max_beauty = max(max_beauty, right - left + 1)
            
        # After iterating through all possible 'right' positions, 'max_beauty'
        # will hold the maximum possible length of a subsequence of equal elements.
        return max_beauty