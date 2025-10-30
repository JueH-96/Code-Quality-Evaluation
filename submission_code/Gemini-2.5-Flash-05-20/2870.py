from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_overall_length = -1
        
        # current_alternating_length:
        # Stores the length of the alternating subarray being actively formed.
        # - If 0, no alternating subarray is currently being tracked.
        # - If > 0, it represents the length of the alternating subarray that ends
        #   at the element `nums[i-1]` from the current loop iteration `i`.
        # It's set to 2 when a new alternating pair (diff=1) is found, or
        # incremented if the pattern continues.
        current_alternating_length = 0 
        
        # last_expected_diff:
        # This variable stores the difference (either 1 or -1) that the next
        # pair of elements (nums[i] - nums[i-1]) should have if the current
        # alternating sequence is to continue.
        # It gets properly set when `current_alternating_length` becomes 2.
        last_expected_diff = 0 # Initial value is irrelevant until used.

        # Iterate from the second element (index 1) to compare each element
        # with its immediate predecessor.
        for i in range(1, n):
            diff = nums[i] - nums[i-1]

            if current_alternating_length == 0:
                # Case 1: No alternating subarray is currently being tracked.
                # Check if the current pair (nums[i-1], nums[i]) can start a new one.
                if diff == 1:
                    # If the difference is 1, a new alternating subarray of length 2 begins.
                    current_alternating_length = 2 
                    max_overall_length = max(max_overall_length, current_alternating_length)
                    # The next expected difference after a +1 should be -1.
                    last_expected_diff = -1 
                # If diff is not 1, current_alternating_length remains 0, meaning no new sequence started.
            else:
                # Case 2: An alternating subarray is currently being tracked.
                # This subarray ends at nums[i-1] and has length `current_alternating_length`.
                # Check if nums[i] extends this existing sequence according to the expected pattern.
                if diff == last_expected_diff:
                    # The current element nums[i] correctly extends the pattern.
                    current_alternating_length += 1 
                    max_overall_length = max(max_overall_length, current_alternating_length)
                    # Toggle the expected difference for the very next step.
                    # (e.g., if expected was -1, next expected will be 1).
                    last_expected_diff *= -1 
                else:
                    # The current alternating sequence is broken.
                    # We now need to check if nums[i-1], nums[i] can start a *new* 
                    # alternating sequence from this point, similar to Case 1.
                    if diff == 1:
                        # If the difference is 1, a new alternating subarray of length 2 starts here.
                        current_alternating_length = 2 
                        max_overall_length = max(max_overall_length, current_alternating_length)
                        # The next expected difference after a +1 should be -1.
                        last_expected_diff = -1 
                    else:
                        # If the difference is not 1, no new alternating subarray starts either.
                        # Reset current_alternating_length to 0, indicating no active sequence.
                        current_alternating_length = 0 
        
        return max_overall_length