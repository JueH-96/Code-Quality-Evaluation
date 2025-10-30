import sys

def is_palindrome(n):
    return str(n) == str(n)[::-1]

def find_nth_palindrome(n):
    count = 0
    number = 0
    while count < n:
        number += 1
        if is_palindrome(number):
            count += 1
    return number

n = int(sys.stdin.readline().strip())
print(find_nth_palindrome(n))