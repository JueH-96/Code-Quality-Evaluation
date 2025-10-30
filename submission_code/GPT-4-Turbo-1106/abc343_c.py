def is_palindrome(num):
    return str(num) == str(num)[::-1]

def max_palindromic_cube(N):
    # Start from the cube root of N and go downwards
    x = int(N**(1/3))
    while True:
        cube = x**3
        if is_palindrome(cube) and cube <= N:
            return cube
        x -= 1

# Read input from stdin
N = int(input().strip())

# Find and print the maximum palindromic cube number not greater than N
print(max_palindromic_cube(N))