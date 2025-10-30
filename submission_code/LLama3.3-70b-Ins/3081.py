from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        # Initialize two pointers, one at the start and one at the end of the array
        left, right = 0, len(nums) - 1
        
        # Continue the process until the two pointers meet
        while left < right:
            # If the element at the left pointer is less than the element at the right pointer
            if nums[left] < nums[right]:
                # Remove the elements at the left and right pointers
                left += 1
                right -= 1
            else:
                # If the elements are equal, break the loop
                break
        
        # The minimum length of the array after removals is the number of elements between the two pointers
        return right - left + 1