# YOUR CODE HERE
def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

n = int(input())
ans = 1
for i in range(1, 1000001):
    cube = i ** 3
    if cube > n:
        break
    if is_palindrome(cube):
        ans = cube
print(ans)