import sys

def longest_palindrome_substring(s):
    n = len(s)
    if n == 0:
        return 0

    # Table to store the palindrome status
    dp = [[False] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    max_length = 1
    for i in range(n):
        dp[i][i] = True

    # Check for sub-string of length 2
    start = 0
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2

    # Check for lengths greater than 2 and less than or equal to n
    for k in range(3, n + 1):
        for i in range(n - k + 1):
            j = i + k - 1
            if dp[i + 1][j - 1] and s[i] == s[j]:
                dp[i][j] = True
                if k > max_length:
                    start = i
                    max_length = k

    return max_length

def main():
    input = sys.stdin.read().strip()
    result = longest_palindrome_substring(input)
    print(result)

if __name__ == "__main__":
    main()