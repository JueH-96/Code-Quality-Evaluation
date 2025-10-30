def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

palindromic_cubes = []
for x in range(1, 1000001):
    cube = x**3
    if is_palindrome(cube):
        palindromic_cubes.append(cube)

palindromic_cubes.sort()

n = int(input())
result = 0
for p in reversed(palindromic_cubes):
    if p <= n:
        result = p
        break
print(result)