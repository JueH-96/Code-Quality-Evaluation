class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Precompute prefix sums for each character
        prefix = [[0] * 26 for _ in range(n + 1)]
        for i in range(1, n + 1):
            prefix[i] = prefix[i-1].copy()
            current_char = s[i-1]
            idx = ord(current_char) - ord('a')
            prefix[i][idx] += 1
        
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters require 0 partitions
        
        for i in range(1, n + 1):
            for j in range(i):
                # Extract counts for each character in substring s[j:i]
                counts = []
                for k in range(26):
                    cnt = prefix[i][k] - prefix[j][k]
                    if cnt > 0:
                        counts.append(cnt)
                # Check if all non-zero counts are the same
                if len(counts) == 0:
                    continue  # Not possible as j < i
                first = counts[0]
                balanced = True
                for cnt in counts[1:]:
                    if cnt != first:
                        balanced = False
                        break
                if balanced:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n]