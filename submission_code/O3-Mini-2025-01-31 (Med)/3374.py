from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        # Initialize the result with the first element (a single element subarray is always alternating)
        total = 1  
        current_length = 1  # current_length tracks the length of the maximal alternating subarray ending at the current index
        
        # Traverse from the second element onward
        for i in range(1, len(nums)):
            # If current and previous elements are alternating, extend the current sequence
            if nums[i] != nums[i - 1]:
                current_length += 1
            else:
                # Else, reset the alternating length to 1 (single element)
                current_length = 1
            # Add the number of alternating subarrays ending at this index (which is the current_length)
            total += current_length
        
        return total