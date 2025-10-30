def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

def find_max_palindromic_cube(N):
    max_cube = 0
    x = 1
    while True:
        cube = x ** 3
        if cube > N:
            break
        if is_palindrome(cube):
            max_cube = cube
        x += 1
    return max_cube

N = int(input())
print(find_max_palindromic_cube(N))