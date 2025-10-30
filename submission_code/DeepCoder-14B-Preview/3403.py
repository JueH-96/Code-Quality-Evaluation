class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute prefix sums for each character
        prefix = [[0] * 26 for _ in range(n + 1)]
        for i in range(1, n + 1):
            for c in range(26):
                prefix[i][c] = prefix[i-1][c]
            current_char = s[i-1]
            idx = ord(current_char) - ord('a')
            prefix[i][idx] += 1
        
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters require 0 partitions
        
        for i in range(1, n + 1):
            for j in range(0, i):
                # Check if substring s[j:i] is balanced
                counts = []
                for c in range(26):
                    cnt = prefix[i][c] - prefix[j][c]
                    if cnt > 0:
                        counts.append(cnt)
                if not counts:
                    continue  # This case shouldn't happen as j < i
                first = counts[0]
                all_same = True
                for cnt in counts[1:]:
                    if cnt != first:
                        all_same = False
                        break
                if all_same:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n]