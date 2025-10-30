from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        """
        Calculates the number of alternating subarrays in a binary array.

        An alternating subarray is one where no two adjacent elements are the same.
        This solution uses a single pass (O(N) time) and constant extra space (O(1)).

        The core idea is to iterate through the array, keeping track of the length
        of the current alternating subarray ending at the current index. For each
        index `i`, the number of new alternating subarrays we find is equal to the
        length of this current streak. We sum these counts for all indices to get
        the total.
        """
        
        # total_count will accumulate the final answer.
        total_count = 0
        
        # current_streak will store the length of the alternating subarray 
        # ending at the current index.
        current_streak = 0
        
        for i in range(len(nums)):
            # Check if the alternating streak is broken. This happens if the current
            # element is the same as the previous one. The check `i > 0` handles
            # the first element of the array.
            if i > 0 and nums[i] == nums[i-1]:
                # A new streak of length 1 starts with the current element.
                current_streak = 1
            else:
                # The streak continues, or it's the first element.
                # In both cases, we extend the current streak by 1.
                current_streak += 1
            
            # The number of alternating subarrays ending at the current index `i`
            # is equal to the length of the current streak. We add this to our total.
            total_count += current_streak
            
        return total_count