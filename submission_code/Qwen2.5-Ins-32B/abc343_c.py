import sys

def is_palindrome(num):
    return str(num) == str(num)[::-1]

def find_max_palindromic_cube(N):
    max_cube = 0
    # Start from the cube root of N and go downwards
    for i in range(int(N**(1/3)), 0, -1):
        cube = i ** 3
        if cube > N:
            continue
        if is_palindrome(cube):
            max_cube = cube
            break
    return max_cube

N = int(input().strip())
print(find_max_palindromic_cube(N))