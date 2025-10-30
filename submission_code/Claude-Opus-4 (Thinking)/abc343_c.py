def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

N = int(input())

x = 1
max_palindromic_cube = 0

while x ** 3 <= N:
    cube = x ** 3
    if is_palindrome(cube):
        max_palindromic_cube = cube
    x += 1

print(max_palindromic_cube)