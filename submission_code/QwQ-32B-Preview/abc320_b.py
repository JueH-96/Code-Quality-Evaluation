def expand(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1

def longest_palindrome_length(s):
    n = len(s)
    if n == 0:
        return 0
    max_length = 1  # At least one character is a palindrome
    
    for i in range(n):
        # Odd length palindrome
        len1 = expand(s, i, i)
        # Even length palindrome
        len2 = expand(s, i, i + 1)
        # Update the maximum length
        max_length = max(max_length, len1, len2)
    
    return max_length

# Read input
S = input().strip()

# Compute and print the maximum palindrome length
print(longest_palindrome_length(S))