# YOUR CODE HERE
import sys

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def max_palindromic_cube(N):
    x = int(N ** (1/3))
    while x > 0:
        cube = x ** 3
        if cube <= N and is_palindrome(cube):
            return cube
        x -= 1
    return 1

# Read input
N = int(sys.stdin.read().strip())

# Print the result
print(max_palindromic_cube(N))