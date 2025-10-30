def is_palindrome(num):
    s = str(num)
    return s == s[::-1]

def find_max_palindromic_cube(N):
    max_palindrome_cube = 0
    x = 1
    while True:
        cube = x ** 3
        if cube > N:
            break
        if is_palindrome(cube):
            max_palindrome_cube = cube
        x += 1
    return max_palindrome_cube

import sys
input = sys.stdin.read
N = int(input().strip())
print(find_max_palindromic_cube(N))