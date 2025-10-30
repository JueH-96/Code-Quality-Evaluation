def is_palindrome(n):
    return str(n) == str(n)[::-1]

def solve(n):
    for i in range(int(n**(1/3)), 0, -1):
        if is_palindrome(i**3):
            return i**3

n = int(input().strip())
print(solve(n))