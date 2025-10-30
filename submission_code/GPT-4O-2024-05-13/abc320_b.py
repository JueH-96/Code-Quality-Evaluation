# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(S):
    n = len(S)
    max_len = 1
    
    for i in range(n):
        for j in range(i, n):
            if is_palindrome(S[i:j+1]):
                max_len = max(max_len, j - i + 1)
    
    return max_len

import sys
input = sys.stdin.read().strip()
print(max_palindrome_length(input))