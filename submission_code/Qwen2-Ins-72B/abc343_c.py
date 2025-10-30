def is_palindrome(n):
    return str(n) == str(n)[::-1]

def solve(N):
    max_palindrome = 0
    for i in range(1, 10**6):
        cube = i**3
        if cube > N:
            break
        if is_palindrome(cube):
            max_palindrome = cube
    return max_palindrome

N = int(input())
print(solve(N))