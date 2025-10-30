class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[[0 for _ in range(k + 1)] for _ in range(n)] for _ in range(n)]
        
        for diff in range(0, n):
            for i in range(0, n - diff):
                j = i + diff
                if diff == 0:
                    for c in range(0, k + 1):
                        dp[i][j][c] = 1
                elif diff == 1:
                    ord_i_val = ord(s[i]) - ord('a')
                    ord_j_val = ord(s[j]) - ord('a')
                    abs_diff_val = abs(ord_i_val - ord_j_val)
                    dist_cost = min(abs_diff_val, 26 - abs_diff_val)
                    for c in range(0, k + 1):
                        if dist_cost <= c:
                            dp[i][j][c] = 2
                        else:
                            dp[i][j][c] = 1
                else:  # diff >= 2
                    ord_i_char_val = ord(s[i]) - ord('a')
                    ord_j_char_val = ord(s[j]) - ord('a')
                    abs_diff_chars_val = abs(ord_i_char_val - ord_j_char_val)
                    dist_cost = min(abs_diff_chars_val, 26 - abs_diff_chars_val)
                    for c in range(0, k + 1):
                        exc_i_val = dp[i + 1][j][c]
                        exc_j_val = dp[i][j - 1][c]
                        max_val = max(exc_i_val, exc_j_val)
                        if c >= dist_cost:
                            pair_val = 2 + dp[i + 1][j - 1][c - dist_cost]
                            max_val = max(max_val, pair_val)
                        dp[i][j][c] = max_val
        
        return dp[0][n - 1][k]