from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # The problem guarantees nums.length >= 2, so the first operation is always possible.
        # This first operation determines the target score for all subsequent operations.
        target_score = nums[0] + nums[1]

        # Initialize the operation count to 1, accounting for the first mandatory operation.
        ops_count = 1

        # Start checking from the third element (index 2). The operations consume elements in pairs.
        # The indices for the operations are (0, 1), (2, 3), (4, 5), and so on.
        # We start checking from the pair that would be at indices (2, 3) after the first operation.
        i = 2

        # Continue checking as long as there are at least two elements remaining in the list
        # starting from the current index `i`. The remaining elements are `nums[i], nums[i+1], ...`.
        # For the next potential operation, we need elements at indices `i` and `i+1`.
        # This requires `i+1` to be a valid index within the original list `nums`, i.e., `i+1 < len(nums)`.
        while i + 1 < len(nums):
            # Calculate the sum of the current pair being considered for an operation.
            # These elements correspond to indices `i` and `i+1` in the original list.
            current_score = nums[i] + nums[i+1]

            # Check if this current operation's score matches the target score established by the first operation.
            if current_score == target_score:
                # If the score matches, this operation is valid according to the condition.
                # Increment the count of successful operations.
                ops_count += 1
                # Move the index `i` forward by 2 positions to point to the start of the next potential pair.
                # The next pair would be at indices `i+2` and `i+3`.
                i += 2
            else:
                # If the sum does not match the target score, we cannot perform this operation.
                # Since all performed operations must have the same score, we must stop here.
                # Any subsequent operations would also need to match the target score.
                break

        # Return the total number of operations performed that all had the same score.
        return ops_count