from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[j] will store the maximum length of a subsequence that sums up to j.
        # Initialize dp table with -1, meaning no subsequence found yet for that sum.
        # dp[0] is initialized to 0, meaning an empty subsequence sums to 0 with length 0.
        dp = [-1] * (target + 1)
        dp[0] = 0

        # Iterate through each number in the input array
        for num in nums:
            # Iterate through the target sums from target down to the current number.
            # We iterate backwards to avoid using the same number multiple times
            # within a single subsequence when processing the same dp array.
            # This ensures each number from 'nums' is considered at most once
            # for forming a subsequence sum (0/1 knapsack like property).
            # The loop range is from 'target' down to 'num'. The lower bound of range is exclusive,
            # so we go down to num-1 to include num.
            for j in range(target, num - 1, -1):
                # If we can form a subsequence sum j - num (i.e., dp[j - num] is not -1)
                if dp[j - num] != -1:
                    # Option 1: Don't include the current number `num`. The max length is the current dp[j].
                    # Option 2: Include the current number `num`. The max length is dp[j - num] + 1.
                    # We take the maximum of the two options.
                    dp[j] = max(dp[j], dp[j - num] + 1)

        # The result is the maximum length for the target sum.
        # If dp[target] is still -1, it means no subsequence sums up to the target.
        return dp[target]