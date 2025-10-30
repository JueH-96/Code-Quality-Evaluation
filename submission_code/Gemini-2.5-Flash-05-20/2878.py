from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # operations_at_start_index[i] stores the number of operations that must start at index i.
        # This array helps in efficiently tracking the effect of operations as the window slides.
        # Its size is N because an operation could theoretically start at any index up to N-k.
        operations_at_start_index = [0] * n 
        
        # current_active_operations_sum represents the total amount by which the current element nums[i]
        # has been effectively reduced by operations that started within the window [i-k+1, i].
        # This acts as a rolling sum of deductions applied to the elements.
        current_active_operations_sum = 0
        
        for i in range(n):
            # Step 1: Remove the contribution of operations that are no longer in the current window.
            # An operation started at index `i-k` affects elements from `i-k` up to `(i-k) + k - 1 = i-1`.
            # So, for `nums[i]`, operations that started at `i-k` are no longer relevant.
            if i - k >= 0:
                current_active_operations_sum -= operations_at_start_index[i - k]
            
            # Step 2: Calculate the value of the current element `nums[i]` after applying all
            # relevant operations that started at previous indices (0 to i-1) and are still active.
            effective_val = nums[i] - current_active_operations_sum
            
            # Step 3: Check for an invalid state: If `effective_val` is negative, it means `nums[i]`
            # has been over-subtracted by previous operations. This is impossible as elements
            # cannot go below zero with the given operation.
            if effective_val < 0:
                return False
            
            # Step 4: If `effective_val` is positive, `nums[i]` still needs to be reduced to zero.
            # According to the greedy strategy, this must be achieved by starting new operations
            # precisely at the current index `i`.
            if effective_val > 0:
                # We can only start a new operation (a subarray of size k) at index `i` if
                # the entire subarray `[i, i+k-1]` is within the bounds of the array `nums`.
                # This means `i + k - 1` must be less than `n`, which simplifies to `i <= n - k`.
                if i > n - k:
                    # If `i` is too large to start a new `k`-sized subarray, and `effective_val`
                    # is still positive, then `nums[i]` cannot be made zero.
                    return False
                
                # If `i` is a valid starting point, we must apply `effective_val` operations
                # starting at `i` to reduce `nums[i]` to zero.
                operations_at_start_index[i] = effective_val
                # Add these newly started operations to our `current_active_operations_sum`
                # as they will affect `nums[i]` and subsequent elements in their k-length window.
                current_active_operations_sum += effective_val
            
            # If `effective_val` is 0, `nums[i]` is already zero. No new operations are needed
            # starting at index `i`. `operations_at_start_index[i]` remains 0 (its initial value)
            # and `current_active_operations_sum` remains unchanged by new operations for this element.
            
        # If the loop completes without returning False, it means all elements could be successfully
        # reduced to zero following the greedy strategy.
        return True