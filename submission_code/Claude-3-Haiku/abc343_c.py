import math

def is_palindrome(num):
    """
    Checks if a number is a palindrome.
    """
    s = str(num)
    return s == s[::-1]

def is_palindromic_cube(num):
    """
    Checks if a number is a palindromic cube.
    """
    if num < 1:
        return False
    
    cube_root = int(num ** (1/3))
    if cube_root ** 3 == num and is_palindrome(num):
        return True
    return False

def find_max_palindromic_cube(N):
    """
    Finds the maximum palindromic cube number not greater than N.
    """
    for i in range(int(N ** (1/3)), 0, -1):
        cube = i ** 3
        if cube <= N and is_palindromic_cube(cube):
            return cube
    return 0

# Read input from stdin
N = int(input())

# Find the maximum palindromic cube number not greater than N
result = find_max_palindromic_cube(N)

# Print the answer to stdout
print(result)