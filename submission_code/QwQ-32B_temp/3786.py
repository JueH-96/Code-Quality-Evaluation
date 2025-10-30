class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute the cost between all pairs of characters
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a = ord(s[i]) - ord('a')
                b = ord(s[j]) - ord('a')
                diff = abs(a - b)
                cost[i][j] = min(diff, 26 - diff)
        
        INF = float('-inf')
        # Initialize DP table: dp[i][j][c] where c ranges from 0 to k
        dp = [[[0] * (k + 1) for _ in range(n)] for __ in range(n)]
        
        # Base case: single character
        for i in range(n):
            for c in range(k + 1):
                dp[i][i][c] = 1
        
        # Fill DP table for substrings of length l from 2 to n
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                current_cost = cost[i][j]
                # Compute for all possible c from 0 to k
                for c in range(k + 1):
                    option1 = INF
                    if c >= current_cost:
                        # The inner substring is from i+1 to j-1
                        inner_val = dp[i + 1][j - 1][c - current_cost] if (i + 1 <= j - 1) else 0
                        option1 = 2 + inner_val
                    # Option2: exclude i, take the best of i+1 to j with c operations
                    option2 = dp[i + 1][j][c] if (i + 1 <= j) else 0
                    # Option3: exclude j, take the best of i to j-1 with c operations
                    option3 = dp[i][j - 1][c] if (i <= j - 1) else 0
                    dp[i][j][c] = max(option1, option2, option3)
        
        return dp[0][n - 1][k]