from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # total_alternating_subarrays will store the final count of all alternating subarrays.
        total_alternating_subarrays = 0
        
        # current_alternating_length keeps track of the length of the 
        # alternating segment that *ends* at the current index.
        # This variable simultaneously represents the count of alternating subarrays 
        # that end precisely at the current index 'i'.
        current_alternating_length = 0

        # Iterate through the array from left to right, considering each element.
        for i in range(n):
            # For the very first element (i=0):
            # It always forms an alternating subarray of length 1 (e.g., [0] or [1]).
            if i == 0:
                current_alternating_length = 1
            # If the current element `nums[i]` is different from the previous one `nums[i-1]`:
            # This means the alternating sequence can be extended.
            # All alternating subarrays that ended at `i-1` can now be extended by `nums[i]`.
            # Plus, `[nums[i]]` itself is a new alternating subarray.
            # So, the count of alternating subarrays ending at `i` is one more than those ending at `i-1`.
            elif nums[i] != nums[i-1]:
                current_alternating_length += 1
            # If the current element `nums[i]` is the same as the previous one `nums[i-1]`:
            # The alternating sequence is broken. Any subarray that included `nums[i-1]` and `nums[i]`
            # would not be alternating.
            # Thus, a new alternating sequence effectively starts at `nums[i]`.
            # The only alternating subarray ending at `i` that continues from `i-1` is just `[nums[i]]` itself.
            else: # nums[i] == nums[i-1]
                current_alternating_length = 1
            
            # Add the count of alternating subarrays ending at the current index `i` 
            # to our running total.
            total_alternating_subarrays += current_alternating_length
        
        return total_alternating_subarrays