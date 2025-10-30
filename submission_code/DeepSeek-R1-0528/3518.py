class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp = [a[3] * x for x in b]
        for k in (2, 1, 0):
            next_dp = [-10**18] * n
            max_right = -10**18
            for i in range(n-1, -1, -1):
                next_dp[i] = a[k] * b[i] + max_right
                if dp[i] > max_right:
                    max_right = dp[i]
            dp = next_dp
        return max(dp)