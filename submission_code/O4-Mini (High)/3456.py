from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[p][t] = max length of a good subsequence ending at index p
        # having used exactly t transitions
        dp = [[0] * (k + 1) for _ in range(n)]
        ans = 0

        for p in range(n):
            # we can always start a new subsequence at p with length 1 and 0 transitions
            dp[p][0] = max(dp[p][0], 1)
            ans = max(ans, dp[p][0])

            v = nums[p]
            # try to extend the subsequence ending at p
            for t in range(k + 1):
                cur_len = dp[p][t]
                if cur_len == 0:
                    continue
                # look at all future positions q > p
                for q in range(p + 1, n):
                    if nums[q] == v:
                        # no new transition
                        if dp[q][t] < cur_len + 1:
                            dp[q][t] = cur_len + 1
                            ans = max(ans, dp[q][t])
                    else:
                        # would incur a transition
                        if t < k and dp[q][t + 1] < cur_len + 1:
                            dp[q][t + 1] = cur_len + 1
                            ans = max(ans, dp[q][t + 1])

        return ans