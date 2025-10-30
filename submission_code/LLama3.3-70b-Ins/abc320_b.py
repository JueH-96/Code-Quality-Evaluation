def is_palindrome(s):
    """Check if a string is a palindrome."""
    return s == s[::-1]

def max_palindrome_length(s):
    """Find the maximum length of a contiguous substring of S that is a palindrome."""
    max_length = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            substring = s[i:j]
            if is_palindrome(substring):
                max_length = max(max_length, len(substring))
    return max_length

# Read input from stdin
s = input().strip()

# Find and print the maximum length of a contiguous substring of S that is a palindrome
print(max_palindrome_length(s))