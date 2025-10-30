def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

N = int(input())
max_palindromic_cube = 0
x = 1

while x * x * x <= N:
    cube = x * x * x
    if is_palindrome(cube):
        max_palindromic_cube = cube
    x += 1

print(max_palindromic_cube)