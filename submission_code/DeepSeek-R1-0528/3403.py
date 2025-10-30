class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            freq = [0] * 26
            max_freq = 0
            distinct_count = 0
            for j in range(i - 1, -1, -1):
                char = s[j]
                idx = ord(char) - ord('a')
                if freq[idx] == 0:
                    distinct_count += 1
                freq[idx] += 1
                if freq[idx] > max_freq:
                    max_freq = freq[idx]
                if max_freq * distinct_count == (i - j):
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n]