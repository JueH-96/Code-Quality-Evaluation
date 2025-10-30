import sys

def is_palindrome(n):
    return str(n) == str(n)[::-1]

def largest_palindromic_cube(n):
    for i in range(n, -1, -1):
        if is_palindrome(i**3):
            return i**3

N = int(sys.stdin.readline().strip())
print(largest_palindromic_cube(N))