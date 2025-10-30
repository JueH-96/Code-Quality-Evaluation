import math
from typing import List

class Solution:
    """
    Finds the length of the longest subarray satisfying the given conditions:
    1. Starts with an even number (nums[l] % 2 == 0).
    2. Alternating parity for adjacent elements (nums[i] % 2 != nums[i+1] % 2 for l <= i < r).
    3. All elements are less than or equal to the threshold (nums[i] <= threshold for l <= i <= r).
    """
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        """
        Calculates the length of the longest alternating subarray using a single pass (O(N) time complexity).

        Args:
            nums: The input list of integers.
            threshold: The maximum allowed value for elements in the subarray.

        Returns:
            The length of the longest valid subarray satisfying the conditions. Returns 0 if no such subarray exists.
        """
        n = len(nums)
        # max_length stores the maximum length found so far across all possible subarrays
        max_length = 0
        # current_length stores the length of the valid alternating subarray *ending* at the current index `i`
        current_length = 0

        # Iterate through the array elements one by one
        for i in range(n):
            
            # --- Step 1: Check the threshold condition for the current element nums[i] ---
            if nums[i] > threshold:
                # If the current element exceeds the threshold, it cannot be part of any valid subarray.
                # This breaks any ongoing valid alternating subarray.
                # Reset the current_length.
                current_length = 0
                # Move to the next element in the array.
                continue 

            # --- Step 2: Determine the current_length based on previous state and current element ---
            # At this point, we know nums[i] <= threshold.

            # Check if we are potentially extending a previously found valid alternating subarray.
            # This requires current_length > 0 (meaning the subarray ending at i-1 was valid)
            # and i > 0 (to access nums[i-1]).
            if i > 0 and current_length > 0:
                
                # We have a potential extension. Now check the alternating parity condition (Condition 2).
                if nums[i] % 2 != nums[i - 1] % 2:
                    # The parity alternates correctly (e.g., previous was even, current is odd, or vice versa).
                    # Since nums[i] also meets the threshold condition, we can extend the current subarray.
                    current_length += 1
                else:
                    # The parity is the same as the previous element (e.g., both even or both odd).
                    # The alternating sequence is broken.
                    # We must check if the current element nums[i] can start a *new* valid subarray.
                    # Check Condition 1: A new valid subarray must start with an even number.
                    if nums[i] % 2 == 0:
                        # nums[i] is even and <= threshold. It can start a new valid subarray of length 1.
                        current_length = 1
                    else:
                        # nums[i] is odd. It cannot start a new valid subarray.
                        # Reset the current length as no valid subarray ends here.
                        current_length = 0
            else:
                # This block is reached if:
                # a) i == 0 (processing the first element)
                # b) current_length == 0 (no valid subarray was ending at index i-1)
                
                # We need to check if the current element nums[i] can start a *new* valid subarray.
                # Check Condition 1: Must start with an even number.
                if nums[i] % 2 == 0:
                    # nums[i] is even and <= threshold. It can start a new valid subarray of length 1.
                    current_length = 1
                else:
                    # nums[i] is odd. It cannot start a valid subarray.
                    # The length of the valid subarray ending here is 0.
                    current_length = 0 # Explicitly set/keep current_length as 0.

            # --- Step 3: Update the overall maximum length ---
            # After calculating the length of the valid alternating subarray ending at the current index `i`,
            # compare it with the maximum length found so far and update if necessary.
            max_length = max(max_length, current_length)

        # After iterating through all elements, max_length holds the length of the longest valid subarray found.
        return max_length