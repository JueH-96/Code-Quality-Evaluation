import math

def is_palindrome(n):
    return str(n) == str(n)[::-1]

def max_palindromic_cube(n):
    max_cube = -1
    for i in range(1, int(math.pow(n, 1/3)) + 2):
        cube = i**3
        if is_palindrome(cube) and cube <= n:
            max_cube = cube
        else:
            break
    return max_cube if max_cube != -1 else 1

n = int(input())
print(max_palindromic_cube(n))