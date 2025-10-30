from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev_ands = {}
        for num in nums:
            current_ands = {}
            # Process previous AND values
            for and_val, count in prev_ands.items():
                new_and = and_val & num
                current_ands[new_and] = current_ands.get(new_and, 0) + count
            # Add the current number as a subarray of length 1
            current_ands[num] = current_ands.get(num, 0) + 1
            # Update total with the count of subarrays ending here that meet the condition
            total += current_ands.get(k, 0)
            # Prepare for next iteration
            prev_ands = current_ands
        return total