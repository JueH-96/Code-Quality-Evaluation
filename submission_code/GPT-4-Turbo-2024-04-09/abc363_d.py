def is_palindrome(num):
    s = str(num)
    return s == s[::-1]

def nth_palindrome(n):
    count = 0
    num = 0
    while True:
        if is_palindrome(num):
            count += 1
            if count == n:
                return num
        num += 1

import sys
input = sys.stdin.read
n = int(input().strip())
print(nth_palindrome(n))