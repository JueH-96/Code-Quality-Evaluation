from typing import List
from collections import defaultdict

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute nxt: nxt[i][v] gives the smallest index >= i with value v, or -1 if none.
        # We'll build nxt for indices 0..n. For convenience, values are 1-indexed (from 1 to 300)
        # nxt[n] is a base state: no index available after n.
        nxt = [[-1] * 301 for _ in range(n+1)]
        # nxt[n] remains all -1.
        for i in range(n-1, -1, -1):
            # make a copy from nxt[i+1]
            nxt[i] = nxt[i+1][:]  # shallow copy is fine since inner items are ints.
            # set the current value at index i (nums[i] is in 1..300)
            nxt[i][nums[i]] = i

        # dp[i] is a dictionary where key = current allowed gap (an integer)
        # and value = maximum length of a subsequence ending at index i with that allowed gap for the next difference.
        # A new element can always start a subsequence. We use allowed gap 300 as an initial "infinity" marker,
        # since maximum absolute difference between any two numbers in [1,300] is at most 299.
        dp = [defaultdict(int) for _ in range(n)]
        # Every index is a valid starting point with length 1.
        for i in range(n):
            dp[i][300] = 1

        res = 1  # global maximum subsequence length

        # Process indices in increasing order
        for i in range(n):
            # current value
            curr_val = nums[i]
            # For each state at index i, try to extend the subsequence.
            for allowed_gap, length in list(dp[i].items()):
                # Determine the range of candidate values that can be chosen next.
                # We need |candidate - curr_val| <= allowed_gap.
                lo = curr_val - allowed_gap
                hi = curr_val + allowed_gap
                if lo < 1:
                    lo = 1
                if hi > 300:
                    hi = 300
                # For each candidate value in the allowed range:
                for candidate in range(lo, hi + 1):
                    new_diff = abs(candidate - curr_val)
                    # We want the next index j (with j > i) such that nums[j] == candidate.
                    # We can get that in O(1) from our precomputed nxt array: look at nxt[i+1][candidate]
                    j = nxt[i+1][candidate] if (i+1 <= n) else -1
                    if j == -1:
                        continue
                    # The new allowed gap becomes the absolute difference that we just used.
                    # We update the state at index j.
                    if dp[j].get(new_diff, 0) < length + 1:
                        dp[j][new_diff] = length + 1
                        if length + 1 > res:
                            res = length + 1
        return res