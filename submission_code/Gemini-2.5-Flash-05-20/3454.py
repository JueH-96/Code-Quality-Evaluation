from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        
        # `operations` will store the total minimum operations.
        operations = 0
        
        # `prev_diff` stores the difference `target[i-1] - nums[i-1]` from the previous element.
        # It's initialized to 0, effectively treating `target[-1] - nums[-1]` as 0.
        # This allows a unified calculation for the first element as well.
        prev_diff = 0 
        
        for i in range(n):
            # Calculate the required difference for the current element.
            # This is the value nums[i] needs to change by to become target[i].
            current_diff = target[i] - nums[i]
            
            # --- Logic for positive adjustments (target[i] > nums[i]) ---
            # If current_diff is positive, it means nums[i] needs to be incremented.
            # An increment operation on nums corresponds to reducing `diff` value.
            # We are essentially building a "histogram" for positive target differences.
            
            # `current_positive_contribution` is the portion of `current_diff` that is positive.
            # If `current_diff` is negative, this is 0.
            current_positive_contribution = max(0, current_diff)
            
            # `prev_positive_contribution` is the portion of `prev_diff` that was positive.
            prev_positive_contribution = max(0, prev_diff)
            
            # If the current positive contribution is greater than the previous positive contribution,
            # it means we need to "start" new increment operations (on nums) to cover this new higher level.
            # These new operations are `current_positive_contribution - prev_positive_contribution`.
            if current_positive_contribution > prev_positive_contribution:
                operations += (current_positive_contribution - prev_positive_contribution)
            
            # --- Logic for negative adjustments (target[i] < nums[i]) ---
            # If current_diff is negative, it means nums[i] needs to be decremented.
            # A decrement operation on nums corresponds to increasing `diff` value.
            # We are building another independent "histogram" for absolute negative target differences.
            
            # `current_negative_contribution` is the absolute value of the negative part of `current_diff`.
            # If `current_diff` is positive, this is 0.
            current_negative_contribution = max(0, -current_diff)
            
            # `prev_negative_contribution` is the absolute value of the negative part of `prev_diff`.
            prev_negative_contribution = max(0, -prev_diff)
            
            # If the current negative contribution is greater than the previous negative contribution,
            # it means we need to "start" new decrement operations (on nums) to cover this new deeper level.
            # These new operations are `current_negative_contribution - prev_negative_contribution`.
            if current_negative_contribution > prev_negative_contribution:
                operations += (current_negative_contribution - prev_negative_contribution)
            
            # Update `prev_diff` for the next iteration.
            prev_diff = current_diff
            
        return operations