# YOUR CODE HERE
import sys

def max_palindrome_length(s):
    n = len(s)
    max_length = 1

    for i in range(n):
        # Check for odd length palindromes
        left, right = i, i
        while left >= 0 and right < n and s[left] == s[right]:
            max_length = max(max_length, right - left + 1)
            left -= 1
            right += 1

        # Check for even length palindromes
        left, right = i, i + 1
        while left >= 0 and right < n and s[left] == s[right]:
            max_length = max(max_length, right - left + 1)
            left -= 1
            right += 1

    return max_length

# Read input from stdin
s = sys.stdin.read().strip()

# Print the result
print(max_palindrome_length(s))