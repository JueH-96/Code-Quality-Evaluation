class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp[i][j] represents max score using first i elements of a and first j elements of b
        dp = [[-float('inf')] * (n + 1) for _ in range(5)]
        
        # Base case: using 0 elements of a gives score 0
        for j in range(n + 1):
            dp[0][j] = 0
        
        # Fill the dp table
        for i in range(1, 5):  # for each element in a (1 to 4)
            for j in range(i, n + 1):  # need at least i elements in b
                # Option 1: don't use b[j-1]
                if j > i:  # only if we have more elements than needed
                    dp[i][j] = dp[i][j-1]
                
                # Option 2: use b[j-1] as the i-th element
                if j >= i:  # we have enough elements
                    dp[i][j] = max(dp[i][j], dp[i-1][j-1] + a[i-1] * b[j-1])
        
        return dp[4][n]