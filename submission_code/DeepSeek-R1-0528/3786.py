class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        cost_table = [[0]*26 for _ in range(26)]
        for a in range(26):
            for b in range(26):
                min_cost = float('inf')
                for x in range(26):
                    d1 = abs(a - x)
                    d1 = min(d1, 26 - d1)
                    d2 = abs(b - x)
                    d2 = min(d2, 26 - d2)
                    min_cost = min(min_cost, d1 + d2)
                cost_table[a][b] = min_cost
        
        dp_prev1 = [[1] * (k+1) for _ in range(n)]
        
        if n == 1:
            return 1
        
        dp_prev2 = None
        
        for L in range(2, n+1):
            dp_curr = [[0] * (k+1) for _ in range(n - L + 1)]
            for i in range(n - L + 1):
                j = i + L - 1
                a_idx = ord(s[i]) - ord('a')
                b_idx = ord(s[j]) - ord('a')
                cost_ij = cost_table[a_idx][b_idx]
                for c in range(k+1):
                    skip_i = dp_prev1[i+1][c]
                    skip_j = dp_prev1[i][c]
                    best = max(skip_i, skip_j)
                    if cost_ij <= c:
                        inner = 0
                        if L > 2:
                            inner = dp_prev2[i+1][c - cost_ij]
                        best = max(best, 2 + inner)
                    dp_curr[i][c] = best
            dp_prev2 = dp_prev1
            dp_prev1 = dp_curr
        
        return dp_prev1[0][k]