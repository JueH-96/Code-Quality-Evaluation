class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        # dp[i][j] will store the longest palindromic subsequence in s[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single letter palindromes
        for i in range(n):
            dp[i][i] = 1
        
        # Fill the dp table
        for length in range(2, n + 1):  # length of the substring
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = 2 + (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0)
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        # Function to calculate minimum changes to make s[i:j+1] a palindrome
        def min_changes_to_palindrome(i, j):
            if i >= j:
                return 0
            if s[i] == s[j]:
                return min_changes_to_palindrome(i + 1, j - 1)
            else:
                return 1 + min(min_changes_to_palindrome(i + 1, j), min_changes_to_palindrome(i, j - 1))
        
        # Try to maximize the length of the palindrome that can be achieved with at most k changes
        max_length = 1  # at least every single character is a palindrome
        for i in range(n):
            for j in range(i, n):
                if min_changes_to_palindrome(i, j) <= k:
                    max_length = max(max_length, dp[i][j])
        
        return max_length