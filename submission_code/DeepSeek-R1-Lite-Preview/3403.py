class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Initialize prefix frequency array
        prefix = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            for c in range(26):
                prefix[i+1][c] = prefix[i][c]
            char_idx = ord(s[i]) - ord('a')
            prefix[i+1][char_idx] += 1
        
        # Initialize dp array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Empty string requires 0 partitions
        
        for i in range(1, n + 1):
            for j in range(i):
                # Get frequency of each character in s[j:i]
                freq = [prefix[i][c] - prefix[j][c] for c in range(26)]
                # Collect non-zero frequencies
                freq_set = set(f for f in freq if f > 0)
                if len(freq_set) <= 1:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]