# YOUR CODE HERE
def is_palindrome(n):
    return str(n) == str(n)[::-1]

def find_max_palindromic_cube(N):
    x = int(N**(1/3))
    while x > 0:
        cube = x**3
        if cube <= N and is_palindrome(cube):
            return cube
        x -= 1
    return 1

N = int(input())
print(find_max_palindromic_cube(N))