class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Precompute prefix sums for each character
        prefix = [[0] * 26 for _ in range(n + 1)]
        for i in range(1, n + 1):
            for c in range(26):
                prefix[i][c] = prefix[i-1][c]
            char = ord(s[i-1]) - ord('a')
            prefix[i][char] += 1
        
        # Initialize DP array with infinity
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string
        
        for i in range(1, n + 1):
            for j in range(i):
                # Check if substring s[j..i-1] is balanced
                counts = []
                for c in range(26):
                    cnt = prefix[i][c] - prefix[j][c]
                    if cnt > 0:
                        counts.append(cnt)
                # Check if all counts are the same
                if not counts:
                    continue  # Empty substring, not possible here
                first = counts[0]
                balanced = all(cnt == first for cnt in counts)
                if balanced:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]