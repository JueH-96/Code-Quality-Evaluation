from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # This problem is a variation of the 0/1 Knapsack problem.
        # We want to find a subsequence of numbers that sums to `target`,
        # and among all such subsequences, we want the one with the maximum
        # number of elements (length).
        #
        # Let dp[j] be the maximum length of a subsequence that sums to j.
        # Our goal is to find dp[target].

        # Initialize a DP array of size `target + 1`.
        # dp[j] is initialized to -1 to indicate that sum `j` is not yet achievable.
        dp = [-1] * (target + 1)

        # Base case: a sum of 0 can be achieved with an empty subsequence,
        # which has a length of 0.
        dp[0] = 0

        # Iterate through each number in the input list.
        for num in nums:
            # To correctly model a subsequence (where each element from the original
            # array can be used at most once), we must iterate the target sums `j`
            # in reverse order. This ensures that when we calculate dp[j],
            # the value dp[j - num] corresponds to the state *before* considering
            # the current `num`.
            for j in range(target, num - 1, -1):
                # If a subsequence summing to `j - num` is possible (i.e., its length is not -1)...
                if dp[j - num] != -1:
                    # ...we can form a new subsequence for sum `j` by including `num`.
                    # The length of this new subsequence would be `dp[j - num] + 1`.
                    # We want the longest subsequence, so we take the maximum of the
                    # current length for `j` and this new potential length.
                    dp[j] = max(dp[j], dp[j - num] + 1)

        # The final answer is the value stored at dp[target]. If it remains -1,
        # it means no subsequence summing to `target` was found.
        return dp[target]