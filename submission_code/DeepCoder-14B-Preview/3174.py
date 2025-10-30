class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute the prefix sums for '0's
        prefix0 = [0] * (n + 1)
        for i in range(n):
            prefix0[i + 1] = prefix0[i] + (1 if s[i] == '0' else 0)
        
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 changes
        
        for i in range(1, n + 1):
            # Try all possible even lengths for the current substring
            for l in range(2, i + 1, 2):
                j = i - l
                if j < 0:
                    continue
                # Calculate the number of '0's in the current substring
                count0 = prefix0[i] - prefix0[j]
                count1 = l - count0  # Number of '1's in the current substring
                cost = min(count0, count1)
                # Update dp[i] with the minimal changes
                if dp[j] + cost < dp[i]:
                    dp[i] = dp[j] + cost
        
        return dp[n]