class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Reverse t for easier comparison
        t_rev = t[::-1]
        m, n = len(s), len(t)
        
        # Create a DP table to store the longest palindromic subsequence lengths
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t_rev[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # The maximum value in the dp table will be the length of the longest palindrome
        max_length = 0
        for i in range(m + 1):
            for j in range(n + 1):
                max_length = max(max_length, dp[i][j])
        
        # Additionally, check for the longest palindromic substring within each string
        def longest_palindrome_substring(x: str) -> int:
            max_len = 0
            length = len(x)
            for center in range(2 * length - 1):
                left = center // 2
                right = left + center % 2
                while left >= 0 and right < length and x[left] == x[right]:
                    max_len = max(max_len, right - left + 1)
                    left -= 1
                    right += 1
            return max_len
        
        max_length = max(max_length, longest_palindrome_substring(s), longest_palindrome_substring(t))
        
        return max_length