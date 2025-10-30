import math
from typing import List

class Solution:
    """
    Finds the length of the shortest special non-empty subarray.
    A subarray is special if the bitwise OR of its elements is at least k.
    """
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum length using nested loops by checking all possible subarrays.

        The algorithm iterates through all possible starting indices `i` and ending indices `j`
        of a subarray `nums[i..j]`. For each subarray, it computes the bitwise OR of its
        elements. If the OR value is greater than or equal to `k`, the length of the
        subarray (`j - i + 1`) is compared with the minimum length found so far, and the
        minimum is updated if necessary. An optimization is included: for a fixed starting
        index `i`, once the condition `OR >= k` is met for `nums[i..j]`, we can stop
        extending the subarray from `i` (break the inner loop), because any longer
        subarray starting at `i` will not yield a shorter length.

        Args:
            nums: A list of non-negative integers (Constraints: 1 <= len(nums) <= 50, 0 <= nums[i] <= 50).
            k: An integer threshold (Constraints: 0 <= k < 64).

        Returns:
            The length of the shortest special subarray, or -1 if no such subarray exists.
            Returns 1 if k is 0, as any single element subarray has OR >= 0.
        """
        n = len(nums)
        
        # Initialize min_length to a value larger than any possible valid length (n).
        # Using float('inf') is a standard practice for minimum calculations.
        # n + 1 would also work since the maximum possible length is n.
        min_length = float('inf')

        # Edge case: if k is 0, any non-empty subarray works. The shortest is length 1.
        # Check if any element is >= k first. If k=0, this condition is always met.
        # The loops below handle k=0 correctly as well, finding length 1.
        # if k == 0:
        #     return 1 # A single element subarray [nums[i]] has OR = nums[i] >= 0.

        # Iterate through all possible start indices 'i' of the subarray.
        for i in range(n):
            # Initialize the bitwise OR for the subarray starting at index 'i'.
            current_or = 0
            # Iterate through all possible end indices 'j' of the subarray, starting from 'i'.
            for j in range(i, n):
                # Expand the subarray nums[i..j] by including nums[j] and update the OR value.
                # The bitwise OR operation ensures that the OR value is non-decreasing as j increases.
                current_or |= nums[j]
                
                # Check if the bitwise OR of the current subarray nums[i..j] meets the threshold k.
                if current_or >= k:
                    # If it meets the condition, calculate the length of this subarray.
                    length = j - i + 1
                    # Update min_length if this subarray is shorter than the shortest one found so far.
                    min_length = min(min_length, length)
                    
                    # Optimization: For a fixed starting index 'i', once we find the 
                    # first (and thus shortest) subarray nums[i..j] that satisfies 
                    # current_or >= k, any longer subarray starting at 'i' 
                    # (i.e., nums[i..j+p] for p > 0) cannot yield a shorter length.
                    # We have found the minimum length for subarrays starting at 'i'.
                    # We can break the inner loop and proceed to check subarrays starting at i+1.
                    break 
        
        # After checking all possible subarrays, if min_length still holds its initial
        # large value (float('inf')), it means no subarray satisfied the condition current_or >= k.
        if min_length == float('inf'):
            # In this case, return -1 as specified.
            return -1
        else:
            # Otherwise, return the minimum length found.
            return min_length