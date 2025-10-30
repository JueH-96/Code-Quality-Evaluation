class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp[i][j] represents the maximum score we can achieve by choosing j indices 
        # from the first i elements of b
        dp = [[-float('inf') for _ in range(5)] for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: no elements chosen
        
        for i in range(1, n + 1):
            dp[i][0] = 0  # We can choose not to pick any elements
            for j in range(1, min(i + 1, 5)):
                # Don't choose the i-th element
                dp[i][j] = dp[i-1][j]
                # Choose the i-th element for the j-th position
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + a[j-1] * b[i-1])
        
        return dp[n][4]