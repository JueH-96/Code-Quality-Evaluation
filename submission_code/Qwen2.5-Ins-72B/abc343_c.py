# YOUR CODE HERE
import math

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def max_palindromic_cube(N):
    max_palindrome = 0
    for x in range(1, int(N**(1/3)) + 1):
        cube = x ** 3
        if cube > N:
            break
        if is_palindrome(cube):
            max_palindrome = max(max_palindrome, cube)
    return max_palindrome

N = int(input())
print(max_palindromic_cube(N))