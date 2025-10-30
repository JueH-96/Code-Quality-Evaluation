from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of all "good" elements in the array.
        An element is good if it is strictly greater than its neighbors at a distance k.
        """
        n = len(nums)
        total_sum = 0
        
        # Iterate through each element and its index in the nums array.
        for i, num in enumerate(nums):
            
            # Check the condition against the left neighbor (at index i - k).
            # This is true if the index is out of bounds or the number is greater.
            # Python's short-circuiting 'or' prevents an IndexError.
            is_good_vs_left = (i - k < 0) or (num > nums[i - k])
            
            # Check the condition against the right neighbor (at index i + k).
            # This is true if the index is out of bounds or the number is greater.
            is_good_vs_right = (i + k >= n) or (num > nums[i + k])
            
            # An element is "good" if it satisfies both the left and right conditions.
            if is_good_vs_left and is_good_vs_right:
                total_sum += num
                
        return total_sum