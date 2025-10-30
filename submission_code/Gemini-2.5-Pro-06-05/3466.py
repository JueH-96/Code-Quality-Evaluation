from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        
        total_count = 0
        # This dictionary stores { and_value: count } for subarrays ending at the previous position.
        prev_ands = {}

        for num in nums:
            # This dictionary will store { and_value: count } for subarrays ending at the current position.
            curr_ands = {}
            
            # Case 1: The subarray consists of only the current number, [num].
            # Its AND value is `num` and there is one such subarray.
            curr_ands[num] = 1

            # Case 2: Extend all subarrays that ended at the previous position with the current number.
            for val, count in prev_ands.items():
                new_and = val & num
                curr_ands[new_and] = curr_ands.get(new_and, 0) + count
            
            # Add the count of subarrays ending at the current position with an AND value of k.
            total_count += curr_ands.get(k, 0)
            
            # The current state becomes the previous state for the next iteration.
            prev_ands = curr_ands
            
        return total_count