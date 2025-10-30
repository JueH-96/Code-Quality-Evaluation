# YOUR CODE HERE
import sys

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def nth_palindrome(n):
    count = 0
    num = 0
    while True:
        if is_palindrome(num):
            count += 1
            if count == n:
                return num
        num += 1

# Read input
n = int(sys.stdin.read().strip())

# Output the result
print(nth_palindrome(n))