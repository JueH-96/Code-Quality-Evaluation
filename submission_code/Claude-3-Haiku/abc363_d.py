def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

def find_nth_palindrome(n):
    if n == 1:
        return 0
    
    count = 1
    i = 1
    while count < n:
        if is_palindrome(i):
            count += 1
        i += 1
    
    return i - 1

import sys
n = int(sys.stdin.readline().strip())
print(find_nth_palindrome(n))