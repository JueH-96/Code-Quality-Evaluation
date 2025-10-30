class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Precompute prefix sums for character counts
        prefix = [[0] * 26 for _ in range(n + 1)]
        for i in range(1, n + 1):
            prefix[i] = prefix[i-1].copy()
            c = s[i-1]
            idx = ord(c) - ord('a')
            prefix[i][idx] += 1
        
        # Initialize DP array with infinity, except dp[0] = 0
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Fill DP array
        for i in range(1, n + 1):
            for j in range(i):
                # Calculate character counts in substring s[j:i]
                count = [prefix[i][k] - prefix[j][k] for k in range(26)]
                # Extract non-zero counts
                non_zero = [x for x in count if x > 0]
                # Check if all non-zero counts are the same
                if not non_zero:
                    continue
                x = non_zero[0]
                if all(y == x for y in non_zero):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n]