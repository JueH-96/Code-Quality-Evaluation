# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def find_palindromic_expression(N):
    str_N = str(N)
    if is_palindrome(str_N):
        return str_N
    
    for i in range(1, len(str_N)):
        if N % i == 0:
            factor1 = str(i)
            factor2 = str(N // i)
            if is_palindrome(factor1 + '*' + factor2 + '*' + factor1):
                return factor1 + '*' + factor2 + '*' + factor1
            if is_palindrome(factor2 + '*' + factor1 + '*' + factor2):
                return factor2 + '*' + factor1 + '*' + factor2
    
    return -1

import sys
input = sys.stdin.read
N = int(input().strip())
print(find_palindromic_expression(N))