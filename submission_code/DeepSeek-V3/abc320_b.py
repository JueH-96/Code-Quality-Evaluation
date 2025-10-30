# YOUR CODE HERE
def longest_palindromic_substring(s):
    n = len(s)
    if n == 0:
        return 0
    dp = [[False for _ in range(n)] for _ in range(n)]
    max_length = 1
    # Every single character is a palindrome
    for i in range(n):
        dp[i][i] = True
    # Check for substring of length 2
    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            max_length = 2
    # Check for substring of length greater than 2
    for length in range(3, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                if length > max_length:
                    max_length = length
    return max_length

S = input().strip()
print(longest_palindromic_substring(S))