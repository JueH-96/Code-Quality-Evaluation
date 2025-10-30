def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(s):
    n = len(s)
    max_len = 1  # Minimum length of palindrome is 1 (single character)
    
    for start in range(n):
        for end in range(start + 1, n + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                max_len = max(max_len, end - start)
    
    return max_len

import sys
input = sys.stdin.read
s = input().strip()
print(max_palindrome_length(s))