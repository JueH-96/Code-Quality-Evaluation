import collections
from typing import List

class Solution:
  def countSubarrays(self, nums: List[int], k: int) -> int:
    total_count = 0
    # freq_prev_step stores {value: count} pairs for subarrays ending at the previous element.
    # value: the bitwise AND sum of such a subarray.
    # count: how many such subarrays (with different start indices) yield this value.
    freq_prev_step = collections.defaultdict(int)

    for x_val in nums:
        # freq_current_step will store {value: count} pairs for subarrays ending at x_val.
        freq_current_step = collections.defaultdict(int)
        
        # Case 1: The subarray consisting of only the current element x_val.
        # Its AND sum is x_val. This is one such subarray.
        freq_current_step[x_val] += 1
        
        # Case 2: Extend subarrays that ended at the previous element.
        # Iterate through each (val, count) pair from freq_prev_step.
        # val: AND sum of a subarray ending at the previous element.
        # count: number of such subarrays.
        for val, count in freq_prev_step.items():
            # Extend these 'count' subarrays by ANDing with x_val.
            new_and_sum = val & x_val
            # Add 'count' to the corresponding new_and_sum in freq_current_step.
            freq_current_step[new_and_sum] += count
        
        # freq_current_step now accurately reflects all AND sums for subarrays ending at x_val.
        # Add the count of subarrays whose AND sum is k to total_count.
        # defaultdict(int) returns 0 if k is not a key, which is the correct behavior.
        total_count += freq_current_step[k]
        
        # For the next iteration, freq_current_step becomes freq_prev_step.
        freq_prev_step = freq_current_step
            
    return total_count