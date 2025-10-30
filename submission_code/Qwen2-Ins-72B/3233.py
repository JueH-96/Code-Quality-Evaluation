from collections import defaultdict

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
            char_count = defaultdict(int)
            for j in range(max(0, i - k), i):
                char_count[s[j]] += 1
                if len(char_count) <= k:
                    dp[i] = min(dp[i], dp[j] + (i - j > 0))
        max_partitions = dp[n]
        
        for i in range(n):
            dp = [0] * (n + 1)
            dp[0] = 0
            for j in range(1, n + 1):
                dp[j] = dp[j - 1] + 1
                char_count = defaultdict(int)
                for l in range(max(0, j - k), j):
                    char_count[s[l]] += 1
                    if len(char_count) <= k:
                        dp[j] = min(dp[j], dp[l] + (j - l > 0))
                if i < j:
                    char_count[s[i]] -= 1
                    if char_count[s[i]] == 0:
                        del char_count[s[i]]
                    char_count['a'] += 1
                    if len(char_count) <= k:
                        dp[j] = min(dp[j], dp[max(0, i - k)] + (j - max(0, i - k) > 0))
            max_partitions = max(max_partitions, dp[n])
        return max_partitions