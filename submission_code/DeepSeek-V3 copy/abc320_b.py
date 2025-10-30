# YOUR CODE HERE
def longest_palindromic_substring(s):
    n = len(s)
    if n == 0:
        return 0
    max_len = 1
    start = 0
    # Initialize a table to store whether a substring is a palindrome
    dp = [[False for _ in range(n)] for _ in range(n)]
    # Every single character is a palindrome
    for i in range(n):
        dp[i][i] = True
    # Check for substring of length 2
    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            max_len = 2
            start = i
    # Check for substring of length greater than 2
    for length in range(3, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                if length > max_len:
                    max_len = length
                    start = i
    return max_len

# Read input
S = input().strip()
# Compute and print the result
print(longest_palindromic_substring(S))