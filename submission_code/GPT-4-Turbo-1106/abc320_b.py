def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(s):
    max_length = 1
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if is_palindrome(s[i:j]):
                max_length = max(max_length, j - i)
    return max_length

# Read input from stdin
S = input().strip()

# Find the maximum length of a contiguous substring that is a palindrome
max_length = max_palindrome_length(S)

# Write the answer to stdout
print(max_length)