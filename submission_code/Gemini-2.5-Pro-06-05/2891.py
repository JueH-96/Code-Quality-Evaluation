from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum possible beauty of the array.

        The problem can be rephrased as finding the largest subset of `nums` where
        the difference between the maximum and minimum element is at most 2*k.
        Any such set of numbers can be made equal to a common target value `T`.

        By sorting the array, this problem becomes finding the longest subarray
        that satisfies this condition. This can be efficiently solved using
        a sliding window approach.
        """
        
        nums.sort()
        
        left = 0
        max_beauty = 0
        
        for right in range(len(nums)):
            # The current window is from index `left` to `right`.
            # The condition for all elements in this window to be made equal is
            # `nums[right] - nums[left] <= 2 * k`.
            
            # If the window is invalid, shrink it from the left by moving
            # the `left` pointer forward.
            while nums[right] - nums[left] > 2 * k:
                left += 1
            
            # The current window `nums[left...right]` is valid. Its length is a
            # candidate for the maximum beauty.
            current_beauty = right - left + 1
            if current_beauty > max_beauty:
                max_beauty = current_beauty
            
        return max_beauty