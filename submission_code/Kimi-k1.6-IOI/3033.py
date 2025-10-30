class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Collect indices where s1 and s2 differ
        diffs = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diffs.append(i)
        
        m = len(diffs)
        if m % 2 != 0:
            return -1
        
        # Initialize DP array where dp[i] is the min cost for the first i elements
        dp = [float('inf')] * (m + 1)
        dp[0] = 0
        
        for i in range(2, m + 1, 2):
            # Cost of pairing the last two elements
            dp[i] = dp[i - 2] + min(x, diffs[i - 1] - diffs[i - 2])
            
            # Check all possible even j to pair with i-1 (0-based in diffs)
            for j in range(0, i - 1, 2):
                if (i - j - 2) >= 0:
                    current_cost = dp[j] + dp[i - j - 2] + min(x, diffs[i - 1] - diffs[j])
                    if current_cost < dp[i]:
                        dp[i] = current_cost
        
        return dp[m]