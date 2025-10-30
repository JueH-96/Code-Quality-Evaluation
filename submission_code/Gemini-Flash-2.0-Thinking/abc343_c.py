def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

n = int(input())
max_palindromic_cube = 0
max_x = int(n**(1/3)) + 2

for x in range(1, max_x + 1):
    cube = x * x * x
    if cube <= n:
        if is_palindrome(cube):
            max_palindromic_cube = max(max_palindromic_cube, cube)
    else:
        break

print(max_palindromic_cube)