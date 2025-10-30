MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        """
        Counts the number of different sets of distinct positive integers such that the sum of
        their x-th powers equals n. The answer is returned modulo 1_000_000_007.
        """
        # Pre–compute all powers that do not exceed n
        powers = []
        base = 1
        while (p := base ** x) <= n:    # walrus operator keeps code compact (Python 3.8+)
            powers.append(p)
            base += 1

        # 0/1 knapsack style DP: dp[s] = number of ways to reach sum s
        dp = [0] * (n + 1)
        dp[0] = 1                       # One way to have sum 0: choose no number

        for p in powers:                # each power can be used at most once
            for s in range(n, p - 1, -1):
                dp[s] = (dp[s] + dp[s - p]) % MOD

        return dp[n]