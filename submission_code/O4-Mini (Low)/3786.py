class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        # dp[i][j][c]: max palindromic subseq len in s[i..j] with exactly c operations
        # We'll only keep dp for current lengths to save memory.
        dp = [[[0] * (k+1) for _ in range(n)] for __ in range(n)]
        
        # Base case: substrings of length 1
        for i in range(n):
            for c in range(k+1):
                dp[i][i][c] = 1
        
        # Precompute cost to match any two chars
        cost = [[0]*26 for _ in range(26)]
        for a in range(26):
            for b in range(26):
                d = abs(a - b)
                cost[a][b] = min(d, 26 - d)
        
        # Build up lengths from 2 to n
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                # For each budget c
                for c_budget in range(k+1):
                    # Option 1: skip s[i]
                    best = dp[i+1][j][c_budget]
                    # Option 2: skip s[j]
                    if dp[i][j-1][c_budget] > best:
                        best = dp[i][j-1][c_budget]
                    # Option 3: pair s[i] and s[j]
                    w = cost[ord(s[i]) - 97][ord(s[j]) - 97]
                    if w <= c_budget:
                        val = 2
                        if i+1 <= j-1:
                            val += dp[i+1][j-1][c_budget - w]
                        if val > best:
                            best = val
                    dp[i][j][c_budget] = best
        
        # Answer is the maximum over budgets up to k
        return max(dp[0][n-1][:k+1])