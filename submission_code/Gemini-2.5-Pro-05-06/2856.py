from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate the number of distinct elements in the whole array.
        # This is the target number of distinct elements for a complete subarray.
        # Using set() on the list is a concise way to find all distinct elements.
        distinct_elements_in_full_array = set(nums)
        target_distinct_count = len(distinct_elements_in_full_array)
        
        complete_subarray_count = 0
        
        # Iterate over all possible start_index of a subarray.
        for start_index in range(n):
            # For each start_index, expand the subarray to the right (end_index).
            # Keep track of distinct elements in the current window nums[start_index ... end_index].
            current_window_distinct_elements = set()
            for end_index in range(start_index, n):
                # Add the new element nums[end_index] to the window.
                current_window_distinct_elements.add(nums[end_index])
                
                # Check if the current window nums[start_index ... end_index] is complete.
                # A window is complete if its number of distinct elements equals
                # the target_distinct_count.
                if len(current_window_distinct_elements) == target_distinct_count:
                    # If nums[start_index ... end_index] is complete,
                    # then any subarray that starts at start_index and ends at end_index or later
                    # (i.e., extends to the right) will also be complete. This is because
                    # extending the subarray cannot decrease the number of distinct elements,
                    # and it cannot exceed target_distinct_count (max possible distinct elements).
                    #
                    # These subarrays are:
                    # nums[start_index ... end_index]
                    # nums[start_index ... end_index+1]
                    # ...
                    # nums[start_index ... n-1]
                    #
                    # The number of such subarrays is (n - 1) - end_index + 1 = n - end_index.
                    # For example, if n=5 and current end_index=3 (0-indexed), 
                    # subarrays ending at index 3 and index 4 are valid.
                    # This is (5 - 3) = 2 subarrays.
                    complete_subarray_count += (n - end_index)
                    
                    # Since we found the smallest end_index for the current start_index
                    # that makes the subarray complete, and all extensions to the right have been counted,
                    # we can break from this inner loop (over end_index)
                    # and move to the next start_index.
                    break
            
        return complete_subarray_count