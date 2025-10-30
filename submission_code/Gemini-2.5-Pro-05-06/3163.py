from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_of_squares = 0

        # Iterate over all possible starting positions for subarrays
        for i in range(n):
            # distinct_elements will store unique numbers for subarrays starting at index i
            # and ending at or after index j.
            # This set is re-initialized for each new starting position i.
            distinct_elements = set()
            
            # Iterate over all possible ending positions for subarrays starting at index i
            for j in range(i, n):
                # The current subarray under consideration is nums[i..j].
                # Add the element nums[j] to our set of distinct elements.
                # If nums[j] is already in the set, the set's content and size do not change.
                # Otherwise, nums[j] is added, and the set's size increases by 1.
                distinct_elements.add(nums[j])
                
                # The number of distinct elements in nums[i..j] is the current size of the set.
                distinct_count = len(distinct_elements)
                
                # Square this distinct count and add it to the total sum.
                total_sum_of_squares += distinct_count * distinct_count
                
        return total_sum_of_squares