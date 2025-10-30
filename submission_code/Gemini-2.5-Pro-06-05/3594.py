import collections
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        """
        Finds the largest potential outlier in an array where n-2 elements are "special",
        one is their sum, and one is an outlier.
        """

        # Let the sum of n-2 special numbers be S.
        # Let the sum element be s_val, and the outlier be o_val.
        # We are given s_val = S.
        # The total sum of the array is total = S + s_val + o_val.
        # Substituting S = s_val, we get total = s_val + s_val + o_val = 2*s_val + o_val.
        # From this, we can find the required sum element for a given outlier:
        # s_val = (total - o_val) / 2.

        total_sum = sum(nums)
        counts = collections.Counter(nums)
        
        # The problem guarantees at least one outlier exists.
        # Initialize with a very small number.
        max_found_outlier = -float('inf')
        
        # We iterate through each unique number in `nums` as a potential outlier.
        # This is more efficient than iterating through the entire list if there are duplicates.
        for potential_outlier in counts:
            numerator = total_sum - potential_outlier
            
            # The numerator must be even for the required sum element to be an integer.
            if numerator % 2 == 0:
                required_sum_element = numerator // 2
                
                # To check if required_sum_element exists in the array at a distinct
                # index, we can temporarily remove one instance of the potential_outlier
                # from our frequency map.
                counts[potential_outlier] -= 1
                
                # Check if the required_sum_element is present in the remaining numbers.
                # counts.get() is used to safely access the count, defaulting to 0.
                if counts.get(required_sum_element, 0) > 0:
                    # If it is, this is a valid configuration, and potential_outlier
                    # is a valid outlier.
                    max_found_outlier = max(max_found_outlier, potential_outlier)
                
                # Restore the count for the next iteration's correctness.
                counts[potential_outlier] += 1
                
        return max_found_outlier