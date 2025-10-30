from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # If already all zeros, no queries are needed.
        if all(x == 0 for x in nums):
            return 0

        # dp[i] is a bitmask where the b-th bit being 1 means
        # we can achieve a total decrement of b at index i.
        dp = [1] * n  # only bit 0 set initially
        # masks[i] has bits [0..nums[i]] set to 1, to truncate shifts
        masks = [(1 << (nums[i] + 1)) - 1 for i in range(n)]

        for idx, (l, r, val) in enumerate(queries):
            # Incorporate this query into each relevant position's DP
            for i in range(l, r + 1):
                # dp[i] |= dp[i] << val, but clipped to sums <= nums[i]
                dp[i] = (dp[i] | (dp[i] << val)) & masks[i]

            # After processing this query, check if every position i
            # can achieve exactly nums[i] total decrement.
            all_zeroable = True
            for i in range(n):
                if ((dp[i] >> nums[i]) & 1) == 0:
                    all_zeroable = False
                    break

            # If yes, the minimum k is idx+1 (1-based count of queries used)
            if all_zeroable:
                return idx + 1

        # If we never reached a state where all positions are zeroable
        return -1