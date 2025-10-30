from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Calculate the total number of distinct elements in the whole array.
        # This is the target distinct count that a "complete" subarray must have.
        target_distinct_count = len(set(nums))
        
        complete_subarrays_count = 0
        
        # Step 2: Iterate through all possible starting indices 'i' of a subarray.
        for i in range(n):
            # Initialize a set to keep track of distinct elements in the current subarray [nums[i]...nums[j]].
            current_subarray_distinct_elements = set()
            
            # Step 3: Iterate through all possible ending indices 'j' for the current starting index 'i'.
            for j in range(i, n):
                # Add the current element nums[j] to the set of distinct elements for the subarray [nums[i]...nums[j]].
                current_subarray_distinct_elements.add(nums[j])
                
                # Check if the number of distinct elements in the current subarray
                # is equal to the target distinct count of the whole array.
                if len(current_subarray_distinct_elements) == target_distinct_count:
                    # If it is, then this subarray [nums[i]...nums[j]] is complete.
                    # Furthermore, any subarray starting at 'i' and ending at 'j', 'j+1', ..., 'n-1'
                    # will also contain all 'target_distinct_count' elements. This is because adding
                    # more elements to the right can only maintain or increase the distinct count,
                    # and it cannot exceed the `target_distinct_count` (which is the maximum possible).
                    
                    # So, all subarrays from [i...j] to [i...n-1] are complete.
                    # The number of such subarrays is (n - 1) - j + 1 = n - j.
                    complete_subarrays_count += (n - j)
                    
                    # Since we found the first (smallest 'j') complete subarray for the current 'i',
                    # we don't need to check further for this 'i' because all subsequent subarrays
                    # starting at 'i' will also be complete.
                    # We can move to the next starting index 'i+1'.
                    break
        
        return complete_subarrays_count