class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # Create a list of indices where s1 and s2 differ
        diff_indices = [i for i in range(n) if s1[i] != s2[i]]
        m = len(diff_indices)
        
        # If the number of differing positions is odd, it's impossible to make them equal
        if m % 2 != 0:
            return -1
        
        # If there are no differing positions, no operations are needed
        if m == 0:
            return 0
        
        # Initialize the dp array
        dp = [float('inf')] * m
        dp[0] = x
        dp[1] = 2
        
        for i in range(2, m):
            # Option 1: Use the first operation on the i-th and some previous j-th position
            dp[i] = dp[i - 2] + x
            # Option 2: Use the second operation on the i-th and (i-1)-th positions
            dp[i] = min(dp[i], dp[i - 1] + 2 * (diff_indices[i] - diff_indices[i - 1] - 1) + 2)
        
        return dp[-1]