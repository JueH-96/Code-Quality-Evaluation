import collections
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total_count = 0
        
        # current_ands is a dictionary (map) that stores the counts of bitwise AND
        # values for all subarrays ending at the current processing position.
        # Key: The bitwise AND value (e.g., 5, 0)
        # Value: The number of subarrays ending at the current position that have this specific AND value (e.g., 2, 1)
        # Example: current_ands = {5: 2, 0: 1} means there are 2 subarrays ending here
        #          whose bitwise AND is 5, and 1 subarray whose bitwise AND is 0.
        current_ands = collections.defaultdict(int) 

        for num in nums:
            # When we process a new number `num` (which is `nums[j]` for current index `j`),
            # we need to calculate the new set of AND values for all subarrays ending at `j`.
            # These values come from two sources:
            # 1. The new single-element subarray `[num]`. Its AND value is `num`.
            # 2. All previous subarrays `nums[i...j-1]` extended by `num`.
            #    If `prev_and_val` was the AND of `nums[i...j-1]`, then the AND of
            #    `nums[i...j]` is `prev_and_val & num`.
            
            # Create a new dictionary to store the AND values and their counts for subarrays
            # that end at the current 'num' (current index).
            new_current_ands = collections.defaultdict(int)

            # Case 1: The subarray consisting only of the current number itself.
            # Its AND value is 'num'. We count it once.
            new_current_ands[num] += 1

            # Case 2: Extend all previously calculated subarrays.
            # Iterate through the (AND value, count) pairs from the 'current_ands' map
            # which represents subarrays ending at the previous index (j-1).
            for prev_and_val, count in current_ands.items():
                # Calculate the new AND value by bitwise ANDing with the current number.
                new_and_val = prev_and_val & num
                # Add the count of subarrays that had 'prev_and_val' to the new_and_val.
                new_current_ands[new_and_val] += count
            
            # After processing all updates for the current 'num', this 'new_current_ands'
            # map becomes the 'current_ands' for the next iteration (for j+1).
            current_ands = new_current_ands
            
            # Check if 'k' is one of the AND values we found for subarrays ending at the current 'num'.
            # If it is, add its count to the total_count. `defaultdict` handles the case where `k` is not a key (returns 0).
            total_count += current_ands[k]
            
        return total_count