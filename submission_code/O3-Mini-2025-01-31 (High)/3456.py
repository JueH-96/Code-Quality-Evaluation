from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][t] will represent the length of the longest good subsequence ending at index i
        # with exactly t transitions (where a transition is when adjacent subsequence elements differ).
        dp = [[0] * (k + 1) for _ in range(n)]
        # A subsequence consisting of a single element has no adjacent differences.
        for i in range(n):
            dp[i][0] = 1

        res = 1  # At least one element can always be chosen.
        # Consider subsequence ending at some index i and try to append an element at j > i.
        for i in range(n):
            for j in range(i + 1, n):
                # If the element at j is the same as at i,
                # then appending it will not add to the transition count.
                if nums[j] == nums[i]:
                    # For every transition count used up to now, update dp[j][t].
                    for t in range(k + 1):
                        if dp[i][t] > 0:
                            new_val = dp[i][t] + 1
                            if new_val > dp[j][t]:
                                dp[j][t] = new_val
                                res = max(res, new_val)
                else:
                    # Different value: appending it uses up one more transition.
                    for t in range(k):  # only t from 0 to k-1 can lead to a valid new transition count.
                        if dp[i][t] > 0:
                            new_val = dp[i][t] + 1
                            if new_val > dp[j][t + 1]:
                                dp[j][t + 1] = new_val
                                res = max(res, new_val)
        return res