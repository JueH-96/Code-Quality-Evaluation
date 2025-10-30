class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            count = {}
            for j in range(i, n):
                char = s[j]
                count[char] = count.get(char, 0) + 1
                # Check if all counts are the same
                first_count = next(iter(count.values()))
                if all(v == first_count for v in count.values()):
                    dp[j+1] = min(dp[j+1], dp[i] + 1)
        
        return dp[n]