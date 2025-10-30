class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n+1)
        dp[0] = 0  # Empty string requires zero substrings
        for i in range(n):
            freq = [0] * 26  # Frequency of characters in current substring
            for j in range(i, n):
                freq[ord(s[j]) - ord('a')] +=1
                # Check if substring s[i..j] is balanced
                counts = [f for f in freq if f > 0]
                if len(set(counts)) == 1:
                    dp[j+1] = min(dp[j+1], dp[i] + 1)
        return dp[n]