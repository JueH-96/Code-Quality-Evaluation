def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

def solve():
    n = int(input())
    ans = 0
    x = 1
    while True:
        cube = x * x * x
        if cube > n:
            break
        if is_palindrome(cube):
            ans = max(ans, cube)
        x += 1
    print(ans)

solve()