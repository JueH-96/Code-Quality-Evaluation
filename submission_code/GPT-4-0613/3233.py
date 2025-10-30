class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        prefix = [0] * (n + 1)
        last = [-1] * 26
        dp = [0] * (n + 1)
        dp[0] = 1
        max_dp = [0] * (n + 1)
        max_dp[0] = 1

        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (1 << (ord(s[i - 1]) - ord('a')))
            dp[i] = max_dp[i - 1]
            for j in range(1, k + 1):
                if i >= j:
                    dp[i] = max(dp[i], dp[i - j] + (bin(prefix[i] ^ (prefix[i - j] if i >= j else 0)).count('1') <= k))
            max_dp[i] = max(max_dp[i - 1], dp[i])

        return max_dp[n]