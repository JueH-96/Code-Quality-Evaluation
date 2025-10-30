class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            freq = {}
            for j in range(i-1, -1, -1):
                char = s[j]
                freq[char] = freq.get(char, 0) + 1
                values = list(freq.values())
                if all(v == values[0] for v in values):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        return dp[n]