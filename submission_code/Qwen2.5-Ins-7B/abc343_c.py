# YOUR CODE HERE
import math

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def find_max_palindromic_cube(N):
    for x in range(int(N**(1/3)), 0, -1):
        cube = x ** 3
        if cube <= N and is_palindrome(cube):
            return cube
    return 1

N = int(input())
print(find_max_palindromic_cube(N))