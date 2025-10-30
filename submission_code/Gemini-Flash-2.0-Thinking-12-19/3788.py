from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # The problem asks for the maximum sum of a unique-element subarray
        # selected from a sequence formed by deleting elements from the original array.
        # This is equivalent to finding the maximum sum of a unique subsequence
        # of the original array. Any unique subsequence [v1, v2, ..., vk]
        # can be formed as a sequence by deleting all other elements. This sequence
        # is itself a unique contiguous subarray of length k.

        # We can use dynamic programming to solve the maximum sum unique subsequence problem.
        # Let dp[val + 100] store the maximum sum of a unique subsequence ending with the value `val`.
        # The values in nums are between -100 and 100. We use an offset of 100 to map them to indices [0, 200].
        
        # Initialize dp array with 0s. dp[val+100] = 0 means the max sum of a unique
        # subsequence ending with `val` is 0 (e.g., starting with an empty sequence
        # before seeing any elements). This also correctly represents the case where
        # a value hasn't been seen yet or the best sequence ending with it is negative
        # and we don't want to consider it.

        # To efficiently find the maximum sum of a unique subsequence from previous elements
        # that *does not* contain the current value `v`, we can maintain the overall
        # maximum sum (`total_max`) and the second maximum sum (`second_total_max`)
        # seen in the `dp` array so far.

        # If the current value `v` corresponds to the value achieving the `total_max`
        # before processing `nums[i]`, then the maximum sum we can append `v` to
        # (from previous unique subsequences not containing `v`) is `second_total_max`.
        # Otherwise, the maximum sum we can append `v` to is `total_max`.

        # The sum we are appending `v` to can also be the sum of the empty subsequence, which is 0.
        # So, the sum to add `v` to is `max(0, max_sum_from_previous_sequences_not_containing_v)`.

        # Let dp[val+100] be the maximum sum of a unique subsequence ending with `val`,
        # considering elements processed so far.
        # Initialize dp array with 0. The value 0 represents the sum of the empty subsequence.
        # The max sum of any unique subsequence seen so far is the max value in dp, which we track
        # with total_max and second_total_max.

        dp_array = [0] * 201
        
        # total_max: the maximum sum of any unique subsequence found so far.
        # second_total_max: the second maximum sum of any unique subsequence found so far.
        # total_max_idx: the index in dp_array corresponding to total_max.
        # Initialize total_max and second_total_max to 0, representing the empty subsequence.
        # total_max_idx can be -1 initially, indicating no specific value holds the max sum yet.

        total_max = 0
        second_total_max = 0
        total_max_idx = -1 # Index corresponding to the value that gives total_max

        for v in nums:
            v_idx = v + 100

            # Max sum of a unique subsequence from previous elements that does NOT contain `v`.
            # This is the maximum value in `dp_array` from before this step, excluding `dp_array[v_idx]`.
            # We find this using total_max and second_total_max from *before* this step.
            
            sum_to_add = 0
            if v_idx == total_max_idx:
                # If the current value `v` was the one achieving the overall max sum,
                # the max sum from previous sequences not containing `v` is the second max sum.
                sum_to_add = second_total_max
            else:
                # If the current value `v` was NOT the one achieving the overall max sum,
                # the max sum from previous sequences not containing `v` is the overall max sum.
                sum_to_add = total_max

            # The maximum sum of a unique subsequence ending with `v` at this step is
            # `v` plus the maximum sum from a previous unique subsequence that does not contain `v`.
            # We also need to consider starting a new subsequence with `v` itself.
            # The sum to add is the max of 0 (empty prefix) and `sum_to_add`.
            # Since dp_array is initialized with 0, `sum_to_add` >= 0 if there was at least one element,
            # and 0 if only negative values were seen or no elements seen yet.
            # The base case of starting with just `v` is implicitly handled if `sum_to_add` is 0 (when total_max was 0).

            candidate_new_dp_v = v + sum_to_add
            
            # Update dp_array[v_idx] with the new maximum sum ending with `v`.
            # This new sum replaces the old maximum sum ending with `v`.
            # Store the old value of dp_array[v_idx] to help update total_max and second_total_max.
            old_dp_v = dp_array[v_idx]
            dp_array[v_idx] = candidate_new_dp_v

            # Update total_max, second_total_max, and total_max_idx based on the new dp_array[v_idx].
            # Need to compare the new value with the old top two maximums.
            
            # If the new value is greater than the current total_max:
            if candidate_new_dp_v > total_max:
                # If the new value is for a different index than the current total_max index:
                # The old total_max becomes the new second_total_max.
                if v_idx != total_max_idx:
                    second_total_max = total_max
                    total_max = candidate_new_dp_v
                    total_max_idx = v_idx
                else:
                    # If the new value is for the same index as the current total_max index,
                    # it just updates the total_max. second_total_max remains unchanged.
                    total_max = candidate_new_dp_v
            # If the new value is not greater than total_max, check if it's greater than second_total_max.
            # It must also be for a different index than the current total_max index.
            elif candidate_new_dp_v > second_total_max and v_idx != total_max_idx:
                 second_total_max = candidate_new_dp_v

        # After processing all elements, the maximum sum of any unique subsequence is total_max.
        # Since the empty subsequence has sum 0 and is always unique, the minimum answer is 0.
        # Our initialization handles this: total_max starts at 0.

        return total_max