def is_palindrome(num):
    return str(num) == str(num)[::-1]

def max_palindromic_cube(N):
    max_cube_root = int(N ** (1/3))  # Calculate the maximum possible cube root
    for x in range(max_cube_root, 0, -1):
        cube = x ** 3
        if cube <= N and is_palindrome(cube):
            return cube
    return None  # This case should not happen as 1^3 is always a palindrome

import sys
input = sys.stdin.read

N = int(input().strip())
result = max_palindromic_cube(N)
print(result)