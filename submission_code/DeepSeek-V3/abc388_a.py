class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[[0] * (k + 1) for _ in range(n)] for __ in range(n)]
        
        for i in range(n):
            for l in range(k + 1):
                dp[i][i][l] = 1
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for l in range(k + 1):
                    if s[i] == s[j]:
                        if length == 2:
                            dp[i][j][l] = 2
                        else:
                            dp[i][j][l] = 2 + dp[i+1][j-1][l]
                    else:
                        # Option 1: skip s[i]
                        val1 = dp[i+1][j][l] if (i+1 <= j) else 0
                        # Option 2: skip s[j]
                        val2 = dp[i][j-1][l] if (i <= j-1) else 0
                        max_val = max(val1, val2)
                        # Option 3: change s[i] and s[j] to match
                        d = abs(ord(s[i]) - ord(s[j]))
                        cost = min(d, 26 - d)
                        if l >= cost:
                            if length == 2:
                                if cost <= l:
                                    max_val = max(max_val, 2)
                            else:
                                if i + 1 <= j - 1:
                                    candidate = 2 + dp[i+1][j-1][l - cost]
                                    max_val = max(max_val, candidate)
                        dp[i][j][l] = max_val
        return dp[0][n-1][k]