from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Helper function to check if a given target_or_value is achievable.
        # This function determines if we can perform at most k operations
        # such that the bitwise OR of the remaining elements is <= target_or_value.
        # This is equivalent to ensuring that for every element `x` in the final array,
        # (x | target_or_value) == target_or_value.
        # i.e., `x` does not have any bit set where `target_or_value` has a 0 bit.
        def check(target_or_value: int) -> bool:
            # `num_segments` counts how many independent segments we can form
            # such that each segment's bitwise AND satisfies the `target_or_value` condition.
            # We want to maximize `num_segments` to minimize operations (ops = N - num_segments).
            
            num_segments = 0
            # Initialize with all bits ON. This acts as an identity for bitwise AND (x & all_ones = x).
            # It also helps signify an empty or just-started segment.
            current_group_and = (1 << 30) - 1  
            
            for x in nums:
                current_group_and &= x  # Accumulate bitwise AND for the current segment
                
                # If the `current_group_and` (the bitwise AND of elements accumulated so far in the current segment)
                # satisfies the `target_or_value` condition:
                # (i.e., `current_group_and` has no bits set that `target_or_value` requires to be 0)
                if (current_group_and | target_or_value) == target_or_value:
                    num_segments += 1  # We successfully formed a valid segment
                    # Reset to start a new segment from the next element.
                    # This greedy choice maximizes the number of segments.
                    current_group_and = (1 << 30) - 1  
            
            # After iterating through all numbers:
            # If `current_group_and` is not `(1 << 30) - 1`, it means the last suffix of the array
            # formed a segment whose AND value is `current_group_and`.
            # This segment was not "cut" because it never satisfied the condition.
            # Since this last segment *must* contribute to the final OR sum, if it itself does not satisfy
            # `(current_group_and | target_or_value) == target_or_value`, then `target_or_value` is not achievable.
            if current_group_and != (1 << 30) - 1:
                return False
            
            # The total number of operations used to get `num_segments` is `n - num_segments`.
            # We need this to be at most `k`.
            # So, `n - num_segments <= k` is equivalent to `num_segments >= n - k`.
            return num_segments >= n - k

        # Binary search for the minimum possible OR value
        # The possible range for the answer is from 0 to 2^30 - 1.
        low = 0
        high = (1 << 30) - 1 # Max possible value for a number given constraints
        ans = high # Initialize ans to the largest possible value

        while low <= high:
            mid = low + (high - low) // 2
            
            if check(mid):
                # If `mid` is achievable, it's a potential answer.
                # Try to find an even smaller achievable value.
                ans = mid
                high = mid - 1
            else:
                # If `mid` is not achievable, we need a larger target_or_value.
                low = mid + 1
        
        return ans