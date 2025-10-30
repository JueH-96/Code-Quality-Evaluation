import math
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum possible bitwise OR of remaining elements
    after applying at most k bitwise AND operations on adjacent elements. Uses binary search
    on the answer combined with a greedy check function.
    """
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        """
        Finds the minimum possible bitwise OR value achievable after performing at most k operations.
        An operation consists of replacing nums[i] and nums[i+1] with nums[i] & nums[i+1].

        Args:
            nums: The list of 0-indexed integers.
            k: The maximum number of operations allowed.

        Returns:
            The minimum possible bitwise OR value of the elements remaining in the array.
        """

        def check(target: int) -> bool:
            """
            Checks if it's possible to achieve a final state where the bitwise OR 
            of all remaining elements is at most `target`, using at most `k` operations.
            
            A final OR value being at most `target` means that every element `v` in the final array 
            must satisfy `v | target == target`. This is because if any element `v` had a bit set 
            that is not set in `target`, the final OR would also have that bit set, making it 
            greater than `target` in terms of bit positions.

            This function calculates the minimum number of operations required to reach such a state
            using a greedy strategy. If this minimum number of operations is less than or equal to k, 
            it returns true.

            The greedy strategy works as follows: Iterate through the numbers, maintaining the bitwise 
            AND (`current_and`) of the elements in the current segment being formed. If adding the 
            current number `num` (by `current_and &= num`) results in `current_and` satisfying the 
            condition `(current_and | target) == target`, it means this segment is valid with respect 
            to the target. We can potentially finalize this segment and start a new one. We reset 
            `current_and` to -1 (all bits set) to indicate the start of a new segment.
            If `current_and` does not satisfy the condition, it means the current segment ending here 
            is invalid. To potentially make it valid later (by ANDing with more elements), the current 
            element *must* be merged with the next element via an AND operation. This forced merge 
            requires one operation. We count this operation by incrementing `ops_needed`.
            
            Args:
                target: The potential maximum OR value to check feasibility for.

            Returns:
                True if an OR value <= target is achievable with at most k operations, False otherwise.
            """
            
            ops_needed = 0
            # Initialize current_and to a value representing all bits set.
            # In Python's 2's complement representation for integers, -1 effectively has all bits set.
            # This acts as the identity element for bitwise AND.
            current_and = -1 
            
            for num in nums:
                # Update the bitwise AND of the current segment incorporating the current number.
                current_and &= num
                
                # Check if the current segment's AND value satisfies the target condition.
                # The condition is that `current_and` should not have any bits set that are 0 in `target`.
                # This is equivalent to `(current_and | target) == target`.
                if (current_and | target) == target:
                    # If condition met, the segment represented by current_and is valid with respect to target.
                    # We can finalize this segment and start a new one from the next element.
                    # Reset current_and to start accumulating the AND for the new segment.
                    current_and = -1
                else:
                    # If condition not met, the current segment ending here is invalid.
                    # To potentially satisfy the condition later by ANDing with future elements,
                    # this element *must* be merged with the following element.
                    # This mandatory merge requires one operation.
                    ops_needed += 1
            
            # After iterating through all numbers, ops_needed holds the minimum number of 
            # operations required to ensure all final segment AND values satisfy the target condition.
            # Return True if this minimum requirement is within the allowed budget k.
            return ops_needed <= k

        # Binary search on the possible answer (the minimum achievable OR value).
        # The minimum OR value can be 0.
        low = 0
        # The maximum possible value for an element is less than 2^30.
        # The maximum possible OR value is also less than 2^30.
        # So, (1 << 30) - 1 is a safe upper bound for the binary search.
        high = (1 << 30) - 1 
        # Variable to store the best result found so far (the minimum valid target).
        # Initialize to the maximum possible value initially.
        ans = high 

        while low <= high:
            # Calculate the middle value to test.
            mid = low + (high - low) // 2
            
            # Check if it's possible to achieve an OR value <= mid.
            if check(mid):
                # If `mid` is achievable, it's a potential answer.
                # Record it and try searching for an even smaller possible value
                # in the lower half of the current search range.
                ans = mid
                high = mid - 1
            else:
                # If `mid` is not achievable, the minimum possible OR value must be larger than `mid`.
                # Search in the upper half of the current search range.
                low = mid + 1
        
        # The loop terminates when low > high. `ans` holds the minimum `target` value
        # for which `check(target)` returned True. This is the minimum possible OR value.
        return ans