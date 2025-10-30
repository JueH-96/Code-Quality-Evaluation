from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # Constraints: 2 <= n <= 10^4.
        # A single element is a valid subsequence of length 1.
        # Any two elements form a subsequence of length 2, which has a single adjacent difference.
        # A sequence of differences of length 1 is non-increasing vacuously.
        # Thus, any subsequence of length 2 is valid. Since n >= 2, a length 2 subsequence always exists.
        # The minimum possible answer is 2 if we only consider subsequences of length >= 2.
        # However, the problem asks for the longest *such* subsequence, implying the condition must hold.
        # A single element has no adjacent differences, so the condition holds trivially. Length 1 is always possible.
        # We initialize max_len to 1 and let the DP find lengths >= 2.

        MAX_VAL = 300 # Max possible value in nums
        # The absolute difference between any two numbers in [1, 300]
        # is between abs(1 - 1) = 0 and abs(1 - 300) = 299.
        # Possible differences are integers from 0 to 299.
        MAX_DIFF = MAX_VAL # We need indices 0 to 299. Array size 300.
        
        # dp[i][diff] stores the length of the longest valid subsequence
        # ending at index i with the last absolute difference being 'diff'.
        # Initialize with 0. A value of 0 means no valid subsequence of length >= 2
        # ends at index i with this specific difference.
        dp = [[0] * MAX_DIFF for _ in range(n)]

        # suffix_max_dp[i][d] stores the maximum value among dp[i][diff]
        # for diff from d to MAX_DIFF - 1.
        # This helps in efficiently finding the max length ending at j
        # with a previous difference greater than or equal to the current difference.
        # Initialize with 0.
        suffix_max_dp = [[0] * MAX_DIFF for _ in range(n)]

        # The minimum possible length is 1 (any single element is a valid subsequence).
        max_len = 1

        # Iterate through the array to build up the DP states
        for i in range(n):
            # For each element at index i, consider all previous elements at index j < i
            for j in range(i):
                current_diff = abs(nums[i] - nums[j])

                # The difference must be a valid index for our diff arrays
                if 0 <= current_diff < MAX_DIFF:
                    current_diff_idx = current_diff

                    # Find the max length of a valid subsequence ending at index j
                    # whose last difference prev_diff satisfies prev_diff >= current_diff.
                    # This max length is efficiently retrieved from suffix_max_dp[j].
                    # suffix_max_dp[j][current_diff_idx] gives the maximum length L (>= 2)
                    # of a valid subsequence ending at j with last diff >= current_diff.
                    # If no such subsequence of length >= 2 exists, this value is 0.
                    max_len_ending_at_j_with_ge_diff = suffix_max_dp[j][current_diff_idx]

                    # Now consider the subsequence ending at index i with nums[j] as the previous element.
                    # The last difference is `current_diff_idx`.
                    # Possible lengths for such a subsequence:
                    # 1. Length 2: Formed by just [nums[j], nums[i]]. This is always valid.
                    # 2. Length > 2: Formed by appending nums[i] to a valid subsequence
                    #    ending at j, say of length L >= 2, with last difference prev_diff >= current_diff.
                    #    The maximum such L is max_len_ending_at_j_with_ge_diff.
                    #    If max_len_ending_at_j_with_ge_diff is L_max >= 2, we can get length L_max + 1.
                    #    If max_len_ending_at_j_with_ge_diff is 0, we cannot extend such a sequence of length >= 2.

                    # The potential length ending at i with diff current_diff_idx by using j is:
                    # max(2, 1 + max_len_ending_at_j_with_ge_diff).
                    # If max_len_ending_at_j_with_ge_diff is 0, this is max(2, 1+0) = 2.
                    # If max_len_ending_at_j_with_ge_diff is L_max >= 2, this is max(2, 1+L_max) = 1+L_max.
                    
                    # Update dp[i][current_diff_idx] by taking the maximum length found so far
                    # for a valid subsequence ending at index i with this specific last difference.
                    dp[i][current_diff_idx] = max(dp[i][current_diff_idx], max(2, 1 + max_len_ending_at_j_with_ge_diff))


            # After computing all dp[i][diff] for the current i using all j < i,
            # compute suffix_max_dp[i].
            # suffix_max_dp[i][d] is the max value in dp[i] from index d onwards.
            # Iterate backwards through possible differences to compute suffix maximums.
            for d in range(MAX_DIFF - 1, -1, -1):
                suffix_max_dp[i][d] = dp[i][d] # Initialize with the value at dp[i][d]
                # If there is a difference d+1, the suffix max up to d must consider the suffix max up to d+1
                if d + 1 < MAX_DIFF:
                    suffix_max_dp[i][d] = max(suffix_max_dp[i][d], suffix_max_dp[i][d+1])

            # Update the overall maximum length found so far.
            # The maximum length of a valid subsequence ending at index i
            # is the maximum value among dp[i][diff] for all diff.
            # This maximum value is efficiently stored in suffix_max_dp[i][0].
            # We update max_len with the maximum length found ending at the current index i.
            # Since max_len is initialized to 1, it correctly represents the minimum possible length
            # (a single element subsequence). If any valid subsequence of length >= 2 is found,
            # suffix_max_dp[i][0] will be >= 2, and max_len will be updated accordingly.
            
            max_len = max(max_len, suffix_max_dp[i][0])

        return max_len