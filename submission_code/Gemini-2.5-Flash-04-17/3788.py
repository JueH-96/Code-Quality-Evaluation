from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)

        # Helper function to find the maximum sum unique subsequence in a subarray, excluding certain values.
        # This DP calculates the max sum unique subsequence by iterating through the subarray.
        # `dp[value]` stores the maximum sum of a unique subsequence ending with `value`.
        # `total_max_sum_so_far` stores the overall maximum sum of any unique subsequence found so far.
        # The transition uses the fact that the max sum of a unique subsequence not ending with `value`
        # (among elements processed so far) is related to `total_max_sum_so_far` and the best sum ending with `value`.
        def get_max_unique_subsequence_sum(sub_arr: List[int], excluded_set_param: set) -> int:
            # max_sum_ending_with[value] = maximum sum of a unique subsequence ending with 'value'
            # using elements from sub_arr processed so far.
            max_sum_ending_with = {}
            
            # total_max_sum_so_far = overall maximum sum of any unique subsequence found so far within sub_arr.
            # Initialized to 0 to represent the empty subsequence.
            total_max_sum_so_far = 0

            for val in sub_arr:
                if val in excluded_set_param:
                    continue

                # To find the new max sum ending with 'val', we append 'val' to a unique subsequence
                # that does *not* contain 'val'. The max sum of such a subsequence before the current 'val'
                # is related to `total_max_sum_so_far` and the best sum ending with 'val' seen previously.
                
                # Let `old_max_sum_ending_with_val` be the maximum sum of a unique subsequence ending with `val`
                # considering elements processed *before* the current `val`.
                old_max_sum_ending_with_val = max_sum_ending_with.get(val, 0)
                
                # The max sum of a unique subsequence using elements processed *before* current `val`,
                # that does *not* end with `val`.
                # This is `total_max_sum_so_far` (max sum before considering the current `val`)
                # minus the maximum sum ending with `val` (before considering the current `val`).
                # If `val` was never seen before, `old_max_sum_ending_with_val` is 0.
                # If `total_max_sum_so_far` was achieved by a sequence NOT ending with `val`,
                # this calculation `total_max_sum_so_far - old_max_sum_ending_with_val` gives the correct sum.
                # If `total_max_sum_so_far` was achieved by a sequence ENDING with `val`,
                # then `total_max_sum_so_far == old_max_sum_ending_with_val`, and the difference is 0,
                # which is correct because the max sum not ending with `val` could be 0 (empty sequence)
                # or potentially another sequence. However, by Kadane's logic applied to unique subsequences,
                # `total_max_sum_so_far` is `max(max_sum_ending_with.values())` (or 0).
                # The correct sum to add to `val` is the max sum of a unique subsequence *before* `val` that *doesn't contain* `val`.
                # This max sum is `total_max_sum_so_far` if `val` hasn't been seen before.
                # If `val` has been seen before, the max sum of a unique subsequence not containing `val`
                # is `total_max_sum_so_far` minus the gain brought by the *last* occurrence of `val`.
                
                # Let's use a simpler DP state: `dp[v]` is the max sum unique subsequence ending with value `v`.
                # The total max sum is `max(0, max(dp.values()))`.
                # When processing `val`, the previous total max sum is `max(0, max(dp[u] for u in dp if u != val))`.
                # This is still O(|values|).

                # The relation `total_max_sum_so_far - max_sum_ending_with.get(val, 0)` gives the max sum of a unique subsequence *not* ending with `val` (among elements processed so far).
                # This is exactly the sum we can append `val` to.
                
                potential_prev_sum = total_max_sum_so_far - max_sum_ending_with.get(val, 0)
                
                # The new max sum ending with the current `val` is `val` plus the max sum of a unique subsequence
                # before this position that does *not* end with `val`. We take `max(0, ...)` because we can always start a new sequence `[val]`.
                new_max_sum_ending_with_val = val + max(0, potential_prev_sum)
                
                # Update the state for `val`. The new max sum ending with `val` is the best we've seen so far ending with `val`.
                max_sum_ending_with[val] = new_max_sum_ending_with_val
                
                # Update the overall maximum sum found so far. It's the max of the previous total max sum
                # and the new max sum ending with `val`.
                total_max_sum_so_far = max(total_max_sum_so_far, new_max_sum_ending_with_val)

            return total_max_sum_so_far # The max sum unique subsequence (can be 0 if all sums are negative)

        # Initialize overall_max_sum with the maximum single element value.
        # A single element subarray is always unique and non-empty. This covers the base case
        # where the optimal unique subarray has length 1.
        overall_max_sum = max(nums)

        # Iterate over all possible pairs of original indices (i, j)
        # such that nums[i] and nums[j] are the first and last elements
        # of the final unique subarray (in original indices).
        # We only consider length > 1 subarrays here. Length 1 is covered by initialization.
        for i in range(n):
            for j in range(i + 1, n):
                # The values nums[i] and nums[j] must be unique in the final subarray.
                # If nums[i] == nums[j], a unique subarray of length > 1 cannot start at original index i and end at original index j.
                if nums[i] == nums[j]:
                    continue

                # The unique subarray must contain nums[i] (from original index i) and nums[j] (from original index j).
                # Any other elements must come from original indices between i and j (i+1 to j-1).
                # These intermediate elements must form a unique subsequence within nums[i+1:j]
                # AND must not have values nums[i] or nums[j] (to maintain uniqueness of the whole subarray).
                
                excluded_set = {nums[i], nums[j]}
                sub_arr = nums[i+1:j]
                
                # Find the maximum sum unique subsequence in nums[i+1:j] excluding nums[i] and nums[j].
                # The helper function returns the max sum, which is >= 0 because the empty subsequence has sum 0,
                # and the DP is initialized to allow this. If all non-empty unique sums are negative, 0 is returned.
                intermediate_max_sum = get_max_unique_subsequence_sum(sub_arr, excluded_set)
                
                # The total sum for this potential unique subarray starting at original index i and ending at original index j
                # is nums[i] + nums[j] + intermediate_max_sum.
                # intermediate_max_sum represents the max sum from the middle part.
                current_pair_sum = nums[i] + nums[j] + intermediate_max_sum
                
                overall_max_sum = max(overall_max_sum, current_pair_sum)

        return overall_max_sum