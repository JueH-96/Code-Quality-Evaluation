from typing import List
import collections

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Store indices for each unique value
        # Use a dictionary where keys are the values from nums
        # and values are lists of indices where that value appears.
        # Since we iterate through nums from left to right, the indices list
        # for each value will be automatically sorted.
        value_indices = collections.defaultdict(list)
        for i, num in enumerate(nums):
            value_indices[num].append(i)
            
        # Initialize the minimum maximum half gap found so far.
        # This will store the minimum of T_v over all values v.
        min_max_half_gap = float('inf')
        
        # Iterate through each unique value and its list of indices.
        for value, indices in value_indices.items():
            # The indices list is sorted: [p_1, p_2, ..., p_k]
            
            max_gap = 0
            k = len(indices) # k is the frequency of the current value
            
            # If a value appears at all indices (k == n), the max_gap will be 1 
            # and max_gap // 2 will be 0. This is handled correctly.
            
            # Calculate the maximum linear gap between consecutive occurrences.
            # This covers gaps (p_2 - p_1), (p_3 - p_2), ..., (p_k - p_{k-1}).
            for i in range(k - 1):
                gap = indices[i+1] - indices[i]
                max_gap = max(max_gap, gap)
                
            # Calculate the wrap-around gap between the last occurrence (p_k)
            # and the first occurrence (p_1).
            # The distance on the circle from index p_k to p_1 is (n - p_k + p_1).
            # The indices involved are p_k, p_k+1, ..., n-1, 0, ..., p_1.
            # The length of this segment (number of steps from p_k to p_1) is (p_1 + n) - p_k.
            wrap_around_gap = (indices[0] + n) - indices[k-1]
            max_gap = max(max_gap, wrap_around_gap)
            
            # The time required for this value to potentially spread everywhere
            # is floor(max_gap / 2).
            time_for_value = max_gap // 2
            
            # Update the overall minimum time required across all values.
            min_max_half_gap = min(min_max_half_gap, time_for_value)
            
        return min_max_half_gap