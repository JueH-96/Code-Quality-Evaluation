class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def max_partitions(s, k):
            if k >= 26:
                return 1
            n = len(s)
            dp = [0] * (n + 1)
            for i in range(n):
                count = [0] * 26
                for j in range(i, n):
                    count[ord(s[j]) - ord('a')] += 1
                    if sum(count) <= k:
                        dp[j + 1] = max(dp[j + 1], dp[i] + 1)
                    else:
                        break
            return dp[n]
        
        return max(max_partitions(s, k), max_partitions(s[:i] + chr(ord('a') + j) + s[i+1:], k) for i in range(len(s)) for j in range(26))