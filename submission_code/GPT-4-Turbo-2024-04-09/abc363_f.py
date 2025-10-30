def is_palindrome(s):
    return s == s[::-1]

def find_palindromic_expressions(N):
    # Check simple direct palindrome without any operations
    str_N = str(N)
    if is_palindrome(str_N) and '0' not in str_N:
        return str_N
    
    # Attempt to construct palindromic expressions using multiplication
    # We will try to find factors of N and see if we can form a palindrome
    # by placing a '*' between them in a way that the result is still a palindrome.
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            a, b = i, N // i
            str_a, str_b = str(a), str(b)
            if '0' in str_a or '0' in str_b:
                continue
            
            # Check combinations of a and b to form a palindrome
            # 1. a*b*a
            candidate1 = str_a + '*' + str_b + '*' + str_a
            if is_palindrome(candidate1):
                return candidate1
            
            # 2. b*a*b
            candidate2 = str_b + '*' + str_a + '*' + str_b
            if is_palindrome(candidate2):
                return candidate2
            
    return -1

import sys
input = sys.stdin.read
N = int(input().strip())
result = find_palindromic_expressions(N)
print(result)