class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(substr: str) -> bool:
            from collections import Counter
            counts = Counter(substr)
            freq = list(counts.values())
            return all(v == freq[0] for v in freq)

        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n]