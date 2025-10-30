class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            counter = {}
            for j in range(i-1, -1, -1):
                c = s[j]
                counter[c] = counter.get(c, 0) + 1
                counts = set(counter.values())
                if len(counts) == 1:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n]