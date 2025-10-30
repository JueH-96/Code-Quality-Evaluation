from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        """
        Determines if an array can be made all zeros by repeatedly decrementing
        subarrays of size k.
        """
        
        # If a subarray of size 1 can be chosen, any non-negative array can be made zero.
        if k == 1:
            return True

        n = len(nums)
        
        # 'carry' represents the cumulative effect of operations on the current element.
        # It's a sliding window sum of the number of operations started in the last k-1 positions.
        carry = 0
        
        for i in range(n):
            # If the current element is smaller than the cumulative decrement `carry`,
            # it's impossible, as we can't perform a negative number of operations (i.e., increment).
            if nums[i] < carry:
                return False
            
            # The number of operations that must start at index `i` is `nums[i] - carry`.
            # An operation starting at `i` is invalid if its subarray `[i...i+k-1]`
            # goes out of bounds. This happens if `i + k > n`. If `nums[i] > carry`
            # for such an `i`, it's impossible. This case is implicitly handled
            # by the final `carry == 0` check.
            
            # We update nums[i] in-place to store the number of operations starting at `i`,
            # which is op_i = nums[i] - carry.
            # Then, we update the carry by adding op_i.
            # A concise way to do this:
            current_ops = nums[i] - carry
            
            # Update carry by adding the effect of current operations.
            carry += current_ops
            
            # Store `current_ops` (op_i) in `nums[i]` to be used for future subtractions from carry.
            nums[i] = current_ops
            
            # If we have processed at least k elements, the operation that started
            # k positions ago (at i-k+1) no longer affects subsequent elements.
            # Its contribution must be removed from the carry.
            if i >= k - 1:
                carry -= nums[i - k + 1]

        # After the loop, the carry must be zero. A non-zero carry implies that
        # some operations would need to start at an index `j` such that `j+k-1 >= n`,
        # which is not possible. The final carry is the sum of these invalid operations.
        return carry == 0