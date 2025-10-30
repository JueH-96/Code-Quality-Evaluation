# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def find_palindrome_expression(N):
    N_str = str(N)
    
    # Check if N itself is a palindrome and does not contain '0'
    if is_palindrome(N_str) and '0' not in N_str:
        return N_str
    
    # Try to find factors of N and construct a palindrome
    for a in range(1, int(N**0.5) + 1):
        if N % a == 0:
            b = N // a
            a_str = str(a)
            b_str = str(b)
            
            # Try to form a palindrome with a and b
            # Form: a*b*a
            if '0' not in a_str and '0' not in b_str:
                candidate = a_str + '*' + b_str + '*' + a_str[::-1]
                if is_palindrome(candidate):
                    return candidate
    
    return -1

import sys
input = sys.stdin.read
N = int(input().strip())
result = find_palindrome_expression(N)
print(result)