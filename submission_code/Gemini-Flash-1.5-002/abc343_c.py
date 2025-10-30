def is_palindrome(n):
    return str(n) == str(n)[::-1]

n = int(input())
ans = 0
i = 1
while i**3 <= n:
    cube = i**3
    if is_palindrome(cube):
        ans = cube
    i += 1
print(ans)