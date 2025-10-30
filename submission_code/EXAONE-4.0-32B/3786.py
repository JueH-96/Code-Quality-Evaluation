class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[[0] * n for _ in range(n)] for __ in range(k+1)]
        
        for r in range(k+1):
            for i in range(n):
                dp[r][i][i] = 1
        
        for r in range(0, k+1):
            for length in range(2, n+1):
                for i in range(0, n - length + 1):
                    j = i + length - 1
                    skip_i = dp[r][i+1][j]
                    skip_j = dp[r][i][j-1]
                    best = max(skip_i, skip_j)
                    
                    diff = abs(ord(s[i]) - ord(s[j]))
                    cost_ij = min(diff, 26 - diff)
                    
                    if r >= cost_ij:
                        inner_r = r - cost_ij
                        if i+1 <= j-1:
                            inner_val = dp[inner_r][i+1][j-1]
                        else:
                            inner_val = 0
                        candidate = 2 + inner_val
                        if candidate > best:
                            best = candidate
                    dp[r][i][j] = best
        
        return dp[k][0][n-1]