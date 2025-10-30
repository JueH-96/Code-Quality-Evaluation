class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            count = {}
            for j in range(i, 0, -1):
                char = s[j - 1]
                count[char] = count.get(char, 0) + 1
                if all(v == count[s[j - 1]] for v in count.values()):
                    dp[i] = min(dp[i], dp[j - 1] + 1)
        
        return dp[n]