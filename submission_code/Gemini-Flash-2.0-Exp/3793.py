class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)
        combined = s + t
        len_combined = len(combined)
        
        dp = [[0] * len_combined for _ in range(len_combined)]
        
        for i in range(len_combined):
            dp[i][i] = 1
        
        max_len = 1
        
        for length in range(2, len_combined + 1):
            for i in range(len_combined - length + 1):
                j = i + length - 1
                if combined[i] == combined[j]:
                    if length == 2:
                        dp[i][j] = 2
                    else:
                        dp[i][j] = dp[i+1][j-1] + 2
                    
                    is_substring_s = False
                    is_substring_t = False
                    
                    if i < n and j >= n:
                        max_len = max(max_len, dp[i][j])
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        
        return max_len